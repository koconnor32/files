import os, json, pickle, time
import numpy as np
import torch
import torch.nn as nn
import rasterio
from rasterio.crs import CRS
import fiona
from shapely.geometry import shape
from shapely.ops import unary_union
import warnings
warnings.filterwarnings("ignore")

BASE_M   = "/scratch.global/ocon0444/peat_modeling"
COV_DIR  = f"{BASE_M}/00_data/covariates_10m"
PATCH_DIR= f"{BASE_M}/00_data/cnn_patches"
MDL_DIR  = f"{BASE_M}/03_models/probability/PROB_CNN_003"
OUT_DIR  = f"{BASE_M}/04_predictions/PROB_CNN_003"
BOUND    = f"{BASE_M}/00_data/boundary/sample_boundary_REGION.shp"
OUT_PATH = f"{OUT_DIR}/cnn_prob_v2_REGION.tif"
REGION   = "RedLake"

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Load config and scaler
with open(f"{MDL_DIR}/cnn_prob_v2_config.json") as f:
    config = json.load(f)
with open(f"{MDL_DIR}/cnn_prob_v2_scaler.pkl", "rb") as f:
    scaler_data = pickle.load(f)
scaler       = scaler_data["scaler"]
tab_features = scaler_data["tab_features"]
n_channels   = config["n_channels"]
n_tab        = config["n_tabular"]
PATCH_SIZE   = config["patch_size"]
HALF         = PATCH_SIZE // 2
print(f"Channels: {n_channels}  Tabular: {n_tab}  Patch: {PATCH_SIZE}x{PATCH_SIZE}")

# Load channel names from patch metadata
with open(f"{PATCH_DIR}/patch_metadata.json") as f:
    meta = json.load(f)
channel_names = meta["channels"]

# Compute per-channel normalization stats from training patches
print("Computing normalization stats from training patches...")
train_patches = np.load(f"{PATCH_DIR}/patches_31x31.npy", mmap_mode="r")
ch_means = np.array([float(train_patches[:,i,:,:].mean()) for i in range(n_channels)], dtype=np.float32)
ch_stds  = np.array([float(train_patches[:,i,:,:].std())  for i in range(n_channels)], dtype=np.float32)
ch_stds  = np.where(ch_stds == 0, 1.0, ch_stds)
del train_patches
print("Stats computed.")

# CNN Architecture (must match training)
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        ]
        if pool: layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)

class DualBranchCNN(nn.Module):
    def __init__(self, n_ch, n_tab):
        super().__init__()
        self.spatial = nn.Sequential(
            ConvBlock(n_ch, 32, pool=True),
            ConvBlock(32,   64, pool=True),
            ConvBlock(64,  128, pool=False),
            nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten(),
        )
        self.tabular_branch = nn.Sequential(
            nn.Linear(n_tab, 128), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64),
            nn.ReLU(inplace=True), nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*4*4 + 64, 256), nn.BatchNorm1d(256),
            nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(256, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(64, 1),
        )
    def forward(self, patch, tab):
        return self.classifier(
            torch.cat([self.spatial(patch), self.tabular_branch(tab)], dim=1)
        ).squeeze(1)

model = DualBranchCNN(n_channels, n_tab).to(device)
model.load_state_dict(torch.load(f"{MDL_DIR}/cnn_prob_v2_final.pt", map_location=device))
model.eval()
print("Model loaded.")

# Region boundary
with fiona.open(BOUND.replace("REGION", REGION)) as src:
    geoms = [shape(f["geometry"]) for f in src]
region_geom = unary_union(geoms)

# Reference raster grid
REF_RAS = f"{COV_DIR}/minnesota_dem_10m.tif"
with rasterio.open(REF_RAS) as ref:
    ref_tf  = ref.transform
    ref_crs = ref.crs
    nrows   = ref.height
    ncols   = ref.width
    bounds  = region_geom.bounds  # minx, miny, maxx, maxy

minx, miny, maxx, maxy = bounds
row_start, col_start = rasterio.transform.rowcol(ref_tf, minx, maxy)
row_end,   col_end   = rasterio.transform.rowcol(ref_tf, maxx, miny)
row_start = max(0, row_start - HALF)
col_start = max(0, col_start - HALF)
row_end   = min(nrows, row_end + HALF)
col_end   = min(ncols, col_end + HALF)
h = row_end - row_start
w = col_end - col_start
out_tf = rasterio.transform.from_bounds(
    *rasterio.transform.xy(ref_tf, row_start, col_start),
    *rasterio.transform.xy(ref_tf, row_end, col_end), w, h)
print(f"Region grid: {h} x {w} = {h*w:,} pixels")

# Raster read helpers
BAND_MAP = {
    "s2_spring_B02":1,"s2_spring_B03":2,"s2_spring_B04":3,"s2_spring_B05":4,
    "s2_spring_B06":5,"s2_spring_B07":6,"s2_spring_B08":7,"s2_spring_B8A":8,
    "s2_spring_B11":9,"s2_spring_B12":10,"s2_spring_NDVI":11,"s2_spring_SWDI":12,
    "s2_summer_B02":1,"s2_summer_B03":2,"s2_summer_B04":3,"s2_summer_B05":4,
    "s2_summer_B06":5,"s2_summer_B07":6,"s2_summer_B08":7,"s2_summer_B8A":8,
    "s2_summer_B11":9,"s2_summer_B12":10,"s2_summer_NDVI":11,"s2_summer_SWDI":12,
    "s2_fall_B02":1,"s2_fall_B03":2,"s2_fall_B04":3,"s2_fall_B05":4,
    "s2_fall_B06":5,"s2_fall_B07":6,"s2_fall_B08":7,"s2_fall_B8A":8,
    "s2_fall_B11":9,"s2_fall_B12":10,"s2_fall_NDVI":11,"s2_fall_SWDI":12,
    "tc_spring_TCB":1,"tc_spring_TCG":2,"tc_spring_TCW":3,
    "tc_summer_TCB":1,"tc_summer_TCG":2,"tc_summer_TCW":3,
    "tc_fall_TCB":1,"tc_fall_TCG":2,"tc_fall_TCW":3,
    "s2_spring_TCB":1,"s2_spring_TCG":2,"s2_spring_TCW":3,
    "s2_summer_TCB":1,"s2_summer_TCG":2,"s2_summer_TCW":3,
}
RASTER_MAP = {
    "s2_spring": f"{COV_DIR}/s2_spring_12bands.tif",
    "s2_summer": f"{COV_DIR}/s2_summer_12bands.tif",
    "s2_fall":   f"{COV_DIR}/s2_fall_12bands.tif",
    "tc_spring": f"{COV_DIR}/tc_spring_merged_5070.tif",
    "tc_summer": f"{COV_DIR}/tc_summer_merged_5070.tif",
    "tc_fall":   f"{COV_DIR}/tc_fall_merged_5070.tif",
}
SINGLE_MAP = {
    "minnesota_dem_10m":               f"{COV_DIR}/minnesota_dem_10m.tif",
    "relativeTopographicPosition_4m":  f"{COV_DIR}/relativeTopographicPosition_4m.tif",
    "relativeTopographicPosition_8m":  f"{COV_DIR}/relativeTopographicPosition_8m.tif",
    "relativeTopographicPosition_16m": f"{COV_DIR}/relativeTopographicPosition_16m.tif",
    "dist_to_water_10m":               f"{COV_DIR}/dist_to_water_10m.tif",
    "dist_to_stream_10m":              f"{COV_DIR}/dist_to_stream_10m.tif",
    "dist_to_road_detailed_10m":       f"{COV_DIR}/dist_to_road_detailed_10m.tif",
    "dist_from_waterbody_edge_10m":    f"{COV_DIR}/dist_from_waterbody_edge_10m.tif",
    "prism_ppt_mn":                    f"{COV_DIR}/prism_ppt_mn.tif",
    "prism_tmax_july_mn":              f"{COV_DIR}/prism_tmax_july_mn.tif",
    "prism_tmean_mn":                  f"{COV_DIR}/prism_tmean_mn.tif",
    "prism_tmin_january_mn":           f"{COV_DIR}/prism_tmin_january_mn.tif",
    "slope":                           f"{COV_DIR}/slope.tif",
    "aspect":                          f"{COV_DIR}/aspect.tif",
    "wetnessIndex":                    f"{COV_DIR}/wetnessIndex.tif",
    "diffFromMeanElev":                f"{COV_DIR}/diffFromMeanElev.tif",
    "devfrommeanelev_4m":              f"{COV_DIR}/devfrommeanelev_4m.tif",
    "devfrommeanelev_8m":              f"{COV_DIR}/devfrommeanelev_8m.tif",
    "devfrommeanelev_16m":             f"{COV_DIR}/devfrommeanelev_16m.tif",
    "planCurvature":                   f"{COV_DIR}/planCurvature.tif",
    "profileCurvature":                f"{COV_DIR}/profileCurvature.tif",
    "maximalCurvature":                f"{COV_DIR}/maximalCurvature.tif",
    "geomorphons":                     f"{COV_DIR}/geomorphons.tif",
    "hillshade":                       f"{COV_DIR}/hillshade.tif",
    "breached_dem":                    f"{COV_DIR}/breached_dem.tif",
    "d8FlowAccumulation":              f"{COV_DIR}/d8FlowAccumulation.tif",
    "dInfFlowAccumulation":            f"{COV_DIR}/dInfFlowAccumulation.tif",
    "mn_nwi_binary":                   f"{COV_DIR}/mn_nwi_cowardin_10m.tif",
    "mn_nwi_merged_1_2":               f"{COV_DIR}/mn_nwi_cowardin_10m.tif",
}

def read_channel(name, row_s, col_s, row_e, col_e):
    win = rasterio.windows.Window(col_s, row_s, col_e-col_s, row_e-row_s)
    if name in ("mn_nwi_binary","mn_nwi_merged_1_2"):
        with rasterio.open(SINGLE_MAP["mn_nwi_binary"]) as src:
            d = src.read(1, window=win).astype(np.float32)
        return ((d==1)|(d==2)).astype(np.float32)
    for prefix in ["s2_spring","s2_summer","s2_fall","tc_spring","tc_summer","tc_fall"]:
        if name.startswith(prefix) and name in BAND_MAP:
            rpath = RASTER_MAP.get(prefix)
            if rpath and os.path.exists(rpath):
                with rasterio.open(rpath) as src:
                    d = src.read(BAND_MAP[name], window=win).astype(np.float32)
                    if src.nodata: d = np.where(d==src.nodata, 0, d)
                return d
    if name in SINGLE_MAP and os.path.exists(SINGLE_MAP[name]):
        with rasterio.open(SINGLE_MAP[name]) as src:
            d = src.read(1, window=win).astype(np.float32)
            if src.nodata: d = np.where(d==src.nodata, 0, d)
        if name == "dist_from_waterbody_edge_10m":
            d = np.where(d==0, 99999, d)
        return d
    return np.zeros((row_e-row_s, col_e-col_s), dtype=np.float32)

def read_tab_feature(name, row_s, col_s, row_e, col_e):
    return read_channel(name, row_s, col_s, row_e, col_e)

# Read all covariate channels for region (with padding for patches)
print("Reading covariate channels...")
pad_rs = max(0, row_start - HALF)
pad_re = min(nrows, row_end + HALF)
pad_cs = max(0, col_start - HALF)
pad_ce = min(ncols, col_end + HALF)
ph = pad_re - pad_rs
pw = pad_ce - pad_cs

all_channels = np.zeros((n_channels, ph, pw), dtype=np.float32)
for ci, ch_name in enumerate(channel_names):
    arr = read_channel(ch_name, pad_rs, pad_cs, pad_re, pad_ce)
    if arr.shape == (ph, pw):
        all_channels[ci] = arr
    # Normalize using training stats
    all_channels[ci] = (all_channels[ci] - ch_means[ci]) / ch_stds[ci]
    all_channels[ci] = np.where(np.isnan(all_channels[ci]), 0, all_channels[ci])
    if (ci+1) % 10 == 0:
        print(f"  Read {ci+1}/{n_channels} channels")
print("All channels loaded.")

# Read tabular features for region
print("Reading tabular features...")
tab_stack = np.zeros((len(tab_features), h, w), dtype=np.float32)
for ti, feat in enumerate(tab_features):
    arr = read_tab_feature(feat, row_start, col_start, row_end, col_end)
    if arr.shape == (h, w):
        tab_stack[ti] = arr
    else:
        tab_stack[ti] = np.nanmedian(arr) if arr.size > 0 else 0
tab_flat = tab_stack.reshape(len(tab_features), -1).T
for fi in range(tab_flat.shape[1]):
    col = tab_flat[:, fi]
    nan_m = np.isnan(col)
    if nan_m.any():
        med = float(np.nanmedian(col))
        tab_flat[nan_m, fi] = med if not np.isnan(med) else 0
tab_scaled = scaler.transform(tab_flat).astype(np.float32)
tab_scaled = np.where(np.isnan(tab_scaled), 0, tab_scaled)
print("Tabular features ready.")

# Sliding window inference
print("Running sliding window inference...")
BATCH_SIZE = 4096
out_prob   = np.full(h * w, -9999, dtype=np.float32)
t0 = time.time()

pixel_indices = []
patch_list    = []
tab_list      = []

for flat_idx in range(h * w):
    row_i = flat_idx // w
    col_i = flat_idx % w
    # Position in padded array
    pr = (row_start + row_i) - pad_rs
    pc = (col_start + col_i) - pad_cs
    if pr < HALF or pr >= ph-HALF or pc < HALF or pc >= pw-HALF:
        out_prob[flat_idx] = -9999
        continue
    patch = all_channels[:, pr-HALF:pr+HALF+1, pc-HALF:pc+HALF+1]
    if patch.shape != (n_channels, PATCH_SIZE, PATCH_SIZE):
        continue
    pixel_indices.append(flat_idx)
    patch_list.append(patch)
    tab_list.append(tab_scaled[flat_idx])

    if len(patch_list) >= BATCH_SIZE:
        pb = torch.FloatTensor(np.stack(patch_list)).to(device)
        tb = torch.FloatTensor(np.stack(tab_list)).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(pb, tb)).cpu().numpy()
        for idx, p in zip(pixel_indices, probs):
            out_prob[idx] = float(p)
        patch_list.clear(); tab_list.clear(); pixel_indices.clear()
        elapsed = round(time.time()-t0, 1)
        done    = flat_idx / (h*w) * 100
        print(f"  {done:.1f}% done  ({elapsed}s)")

# Final batch
if patch_list:
    pb = torch.FloatTensor(np.stack(patch_list)).to(device)
    tb = torch.FloatTensor(np.stack(tab_list)).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(pb, tb)).cpu().numpy()
    for idx, p in zip(pixel_indices, probs):
        out_prob[idx] = float(p)

print(f"Inference done. Elapsed: {round(time.time()-t0,1)}s")

# Save
out_tf2 = rasterio.transform.from_origin(
    rasterio.transform.xy(ref_tf, row_start, col_start)[0],
    rasterio.transform.xy(ref_tf, row_start, col_start)[1],
    10, 10)
prof = {
    "driver":"GTiff","dtype":"float32","count":1,
    "height":h,"width":w,"crs":ref_crs,
    "transform":out_tf2,"nodata":-9999,"compress":"lzw"
}
out_path = OUT_PATH.replace("REGION", REGION)
with rasterio.open(out_path, "w", **prof) as dst:
    dst.write(out_prob.reshape(h, w), 1)

valid = out_prob[out_prob != -9999]
print(f"Saved: {out_path}")
print(f"Valid pixels: {len(valid):,}  mean={valid.mean():.3f}  range=[{valid.min():.3f},{valid.max():.3f}]")
