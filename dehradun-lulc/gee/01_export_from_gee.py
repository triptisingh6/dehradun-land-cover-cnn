"""
STEP 1 — Get the satellite photo + the answer key, cut it into flashcards.

HOW TO RUN THIS:
1. Go to https://colab.research.google.com and create a new notebook.
2. Paste this whole file into a cell (or upload it and %run it).
3. You need a free Google Earth Engine account: https://code.earthengine.google.com/register
4. Run the cell. It will ask you to log in with Google and paste an
   authentication code the first time. After that it remembers you.

WHAT THIS SCRIPT PRODUCES (all saved to your Google Drive, in a folder
called "dehradun_lulc"):
  - dehradun_2016_composite.tif   -> the full winter-2016 photo
  - dehradun_2025_composite.tif   -> the full winter-2025 photo
  - training_patches.tfrecord.gz  -> thousands of labeled flashcards for training
  - training_patches_mixer.json   -> a small file GEE needs alongside the tfrecord

Exports run in the CLOUD (Earth Engine's servers), not on your laptop.
They can take 10-40 minutes. Check progress at:
  https://code.earthengine.google.com/tasks
"""

import ee

# ---------------------------------------------------------------------------
# 0. LOG IN
# ---------------------------------------------------------------------------
ee.Authenticate()          # first time only: opens a login flow
ee.Initialize(project="YOUR-GEE-PROJECT-ID")   # <-- replace with your GEE cloud project id
# (Earth Engine now requires every account to be linked to a Google Cloud
#  project — the registration flow at the link above creates one for you
#  automatically and will show you its ID.)

# ---------------------------------------------------------------------------
# 1. DEFINE THE STUDY AREA — Dehradun / Doon Valley
# ---------------------------------------------------------------------------
# A rectangle covering Dehradun city, the Rispana/Song river corridor,
# nearby forest fringe (north) and agricultural valley floor (south/east).
aoi = ee.Geometry.Rectangle([77.90, 30.20, 78.20, 30.45])

# ---------------------------------------------------------------------------
# 2. CLOUD-MASKED SENTINEL-2 COMPOSITE (reusable function for any date range)
# ---------------------------------------------------------------------------
def get_s2_composite(start_date, end_date, region, max_cloud=20):
    """Returns a single clean, cloud-free-ish Sentinel-2 image for the
    given date range, made by blending (median-ing) many individual photos
    so clouds in any one photo get 'voted out'."""

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(region)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud)))

    s2_clouds = (ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
                 .filterBounds(region)
                 .filterDate(start_date, end_date))

    # Join each image to its matching cloud-probability image
    joined = ee.Join.saveFirst("cloud_mask").apply(
        primary=s2,
        secondary=s2_clouds,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index")
    )

    def mask_clouds(img):
        clouds = ee.Image(img.get("cloud_mask")).select("probability")
        is_not_cloud = clouds.lt(40)  # pixels with <40% cloud probability
        return img.updateMask(is_not_cloud).divide(10000)  # scale reflectance to 0-1

    masked = ee.ImageCollection(joined).map(mask_clouds)

    composite = masked.median().clip(region)
    # Bands we care about: Blue, Green, Red, NIR, SWIR1, SWIR2
    return composite.select(["B2", "B3", "B4", "B8", "B11", "B12"])


composite_2016 = get_s2_composite("2015-11-01", "2016-02-28", aoi)
composite_2020 = get_s2_composite("2019-11-01", "2020-02-28", aoi)
composite_2025 = get_s2_composite("2024-11-01", "2025-02-28", aoi)

# ---------------------------------------------------------------------------
# 3. THE "ANSWER KEY" — ESA WorldCover, remapped to our 5 classes
# ---------------------------------------------------------------------------
# WorldCover 2021 original classes (code: name):
#  10 Tree cover | 20 Shrubland | 30 Grassland | 40 Cropland | 50 Built-up
#  60 Bare/sparse veg | 70 Snow/ice | 80 Water | 90 Wetland | 95 Mangroves | 100 Moss/lichen
worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().clip(aoi)

# Our 5 target classes:
#   0 = Vegetation (forest/shrub/grass)
#   1 = Water
#   2 = Urban / built-up
#   3 = Bare soil
#   4 = Agriculture (cropland)
from_codes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
to_codes   = [0,  0,  0,  4,  2,  3,  3,  1,  1,  0,  3]

labels = worldcover.remap(from_codes, to_codes).rename("landcover")

# ---------------------------------------------------------------------------
# 4. GENERATE FLASHCARDS: stratified sample points, one per class, then
#    cut a small image patch around each point from the 2016 composite.
# ---------------------------------------------------------------------------
# NOTE: originally set to 600 points/64px patches, but this hit GEE's
# "Image.reduceRegions: computed value is too large" error. Reduced to the
# values below, which run reliably. Patch size ends up as (2*radius)+1 = 33.
POINTS_PER_CLASS = 300
PATCH_RADIUS = 16        # -> 33x33 pixel patches ~ 330m x 330m at 10m resolution

sample_points = labels.stratifiedSample(
    numPoints=POINTS_PER_CLASS,
    classBand="landcover",
    region=aoi,
    scale=10,
    seed=42,
    geometries=True
)

# Stack the composite bands + the label band together, then cut a small
# square "neighborhood" around every sample point.
stack = composite_2016.addBands(labels)

kernel = ee.Kernel.square(radius=PATCH_RADIUS, units="pixels")
patches_image = stack.neighborhoodToArray(kernel)

training_patches = patches_image.sampleRegions(
    collection=sample_points,
    scale=10,
    tileScale=16   # higher tileScale = smaller compute chunks = avoids the error above
)

# ---------------------------------------------------------------------------
# 5. EXPORT EVERYTHING TO GOOGLE DRIVE
# ---------------------------------------------------------------------------
# 5a. The flashcards (for training the CNN)
export_patches = ee.batch.Export.table.toDrive(
    collection=training_patches,
    description="dehradun_training_patches",
    folder="dehradun_lulc",
    fileNamePrefix="training_patches",
    fileFormat="TFRecord"
)
export_patches.start()

# 5b. Full composite images (so we can later color in the ENTIRE map, and
#     view them in QGIS)
export_2016 = ee.batch.Export.image.toDrive(
    image=composite_2016,
    description="dehradun_2016_composite",
    folder="dehradun_lulc",
    fileNamePrefix="dehradun_2016_composite",
    region=aoi,
    scale=10,
    maxPixels=1e9
)
export_2016.start()

export_2020 = ee.batch.Export.image.toDrive(
    image=composite_2020,
    description="dehradun_2020_composite",
    folder="dehradun_lulc",
    fileNamePrefix="dehradun_2020_composite",
    region=aoi,
    scale=10,
    maxPixels=1e9
)
export_2020.start()

export_2025 = ee.batch.Export.image.toDrive(
    image=composite_2025,
    description="dehradun_2025_composite",
    folder="dehradun_lulc",
    fileNamePrefix="dehradun_2025_composite",
    region=aoi,
    scale=10,
    maxPixels=1e9
)
export_2025.start()

print("Four export tasks started.")
print("Watch progress here: https://code.earthengine.google.com/tasks")
print("When all three show 'COMPLETED', check your Google Drive folder 'dehradun_lulc'.")
