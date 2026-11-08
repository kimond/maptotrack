# maptotrack

Pick points on a map → get an Assetto Corsa track: a road that follows real streets and real elevation, with terrain, buildings, streets and barriers taken from OpenStreetMap. Fine-tune the result in the app, then in Blender. The app writes the `.kn5` track file itself (no ksEditor needed), both straight from the generator and from your edited Blender scene or any FBX.

```
 Map UI (browser)  →  generator (Python)  →  name.kn5 + ac_track/<name>/ folder  →  Install  →  content/tracks
  waypoints            road / terrain /        (ui, map.png, map.ini, surfaces.ini)
  per-segment mode     buildings / walls
                            └→ track.blend / track.fbx  →  fine-tune in Blender  →  Build KN5 (Blender in background)
```

## Requirements

| Tool | Why | Get it |
|---|---|---|
| Python 3.10+ | runs the app | python.org (tick “Add to PATH”) |
| Blender 4.2+ | fine-tuning, and converting an edited scene / FBX to KN5 (optional) | blender.org |
| Blender **5.0.x** and Studio 397's *rFactor 2 GMT exporter* add-on zip | rFactor 2 export only (optional) | download.blender.org/release/Blender5.0 (the portable `blender-5.0.x-windows-x64.zip` is enough); the add-on zip from Studio 397 (`io_rfactor2_gmt_*.zip`) |
| Internet | routing, OSM, elevation | — |

## Start

Get the code (or download the ZIP from GitHub):

```
git clone https://github.com/kimond/maptotrack.git
cd maptotrack
```

Double-click **`start.bat`** (Windows) or run `./start.sh`. The first run installs packages into `.venv`. The app then opens at <http://127.0.0.1:8765>.

Open ⚙ **Settings** to set the path to `blender.exe`, your `assettocorsa\content\tracks` folder, and your author name.

## Workflow

1. **Location & route**: search for a place and click the map to drop waypoints. Or switch to **Area (free roam)** to make every street in a rectangle or polygon drivable (see **Area mode** below).
   * Each segment can **follow roads** (blue, routed with OSRM) or be a **free line** (orange). Click a segment pill in the list to switch it.
   * Drag a marker to move it and right-click it to delete it.
   * **Add a point in the middle** of the track: click on the track itself (anything within ~14 px of it counts), drag one of the small **+** handles shown half-way along each segment (zoom in to see them on short segments), or press **+** next to a point in the list to insert one half-way to the next point. Clicking away from the track still adds a point at the end. *Undo* reverts any of these.
   * After generating, **right-click** the track to check or set the road width (left-click now adds a point).
   * *Closed circuit* joins the last point back to the first. *Ignore one-way streets* lets routing use one-way streets in both directions, which is useful for street circuits.
   * You can also **import** a KML file (Google My Maps → ⋮ → Export to KML), a GPX file, or a **Google Maps directions link**.
   * **Map layers** (top right of the map): *Map* (OpenStreetMap), *Satellite (Esri)*, and *Aerial photo: Saguenay–Lac-Saint-Jean 2025 (20 cm)*, the same photo the generator reads. It switches on by itself the first time you look at an area it covers; the *Photo* slider (bottom left) fades it so you can see the streets underneath. The photo is fetched once and kept in `cache/imagery`, so panning back is instant.
2. **Road**: set width, width smoothing (evens out the photo-measured width), shoulder, corner smoothing (rounds off 90° junctions), elevation smoothing (removes DEM noise), crown, start-line offset, concrete barriers, and grid/pit slots.
   **Preview road on map** builds only the road (centreline, width measured on the photo along the track, smoothing, manual widths, elevation, start and pit spots) and draws it on the map in a few seconds: the asphalt, its edges (green = clear on the photo, orange = faint), shoulder and barrier lines, with width and elevation charts. Tick **Live** to refresh it by itself when you move a point or change a road setting. It uses the same code as a full generation, so the road you see is the road that gets built (LiDAR heights only come with the full generation).
3. **Scenery**: set the area around the track and the terrain resolution, and turn buildings and other streets on or off.
   **Ground texture**:
   * *Auto* lays the real aerial photo on the ground where a source covers the area (built in: Quebec Saguenay–Lac-Saint-Jean 2025, 20 cm). *Esri World Imagery* works worldwide but is for personal use only.
   * *Land cover colours* paints the ground instead of showing the photo. When a photo exists, the photo is only used to find each surface: lawn, tall/wild grass, dry grass, forest floor, sand, bare soil, gravel and paved areas. Each surface then gets a clean, saturated colour (bright green lawns, olive meadows, dark forest floor, tan sand, dark asphalt) with a little procedural grain, so the surfaces contrast clearly and there are no cars, shadows or blur. Tiny patches (< 3–8 m²) blend into the surface around them. Driveways and parking lots are painted on top with crisp edges, as dark asphalt or light concrete. You can change the colours in `GROUND_STYLE` in `app/landcover.py`. Without a photo it uses a standard palette from the land cover map.
   * *Plain grass* is one tiled grass texture.
   **Trees & bushes** are scattered in forest and shrub areas (ESA WorldCover land cover map + OpenStreetMap woods/parks, plus individual OSM street trees). They're kept clear of the road, barriers, streets and buildings.
   **Read from the aerial photo** (where a photo is available):
   * *Road width → Measure on photo* finds the real asphalt edges on the aerial photo every 2 m (see **How the road width is measured** below) and builds the road between them, so off-centre OpenStreetMap lines are corrected too. Where the photo can't be read, it uses the fixed *Width*.
   * *Buildings* finds the buildings that are missing from OpenStreetMap/Overture, snaps mapped outlines that are 1–3 m off onto the photo, and (with LiDAR) gives every building its measured height. See **How buildings are found** below.
   * *Trees & bushes* are placed where the photo (and LiDAR) shows canopy: every tree, bush and hedge, with evergreens told apart from broadleaf trees. The random scattering is only used where there is no photo. See **Vegetation** below.
   * *Driveways & parking*: neutral-grey, very smooth surfaces that touch a street or a building become drivable asphalt patches with the ground texture. Sand and dry grass (yellowish), tree shadows (textured) and gravel shoulders running along the road are rejected. On the Saguenay circuit this found 63 driveways/lots (7,400 m²); the old colour rule had flagged 828 areas (92,000 m²), mostly sand and shadows. Check the grey shapes on the map. The blobs found on the photo are then cleaned up (`app/driveways.py`): holes are filled (no bush, car or shadow left in the middle), a single driveway becomes a straight rectangle, square to its street or along its own direction when it meets the street at an angle, as wide as it typically is, and it is extended up to 7 m to run 1 m into the street or track (it always touches the asphalt). Parking lots become a rectangle along the street, or a shape with straight edges plus an apron to the street. Blobs that only touch a house (behind it) keep their place, with straight edges. Buildings are cut out. Nothing grows on a driveway (trees, bushes, grass tufts), and no pole or sign stands on one. In the 30 m beside the track the driveway outlines are vertices of the ground, so their edges are straight lines.
   * *Tidy ground*: the surfaces read on the photo are simplified the way a groundskeeper would leave them: borders smoothed by a majority vote within 1.25 m, sand / gravel / soil / dry grass kept only as real areas (at least 4 m wide and 60 m², none within 3 m of a house), so the light fringes round driveways, along kerbs and under eaves become lawn, and patches under 20 m² of lawn or wild grass take the surface around them. Grass tufts follow the result: short tufts on lawn, tall clumps on wild / dry grass, none on sand, gravel, soil or driveways.
   * Roof colours: house roofs take a shingle colour (black, charcoal, grey, brown or red) from how dark or warm their roof looks on the photo compared with the others.
   * *Quebec LiDAR 1 m* (off by default): downloads the government's laser-survey terrain and canopy models once per 1:20 000 map sheet (roughly 0.3–1.5 GB). It gives real ground and road heights instead of 10–30 m data, and trees at their measured positions and heights.
   * *Canada HRDEM 1 m* (on by default, Canada only): Natural Resources Canada's national LiDAR terrain mosaic (HRDEM, 1 m; 2 m when the area is over ~40 km²). Only the window around the track is read from the online Cloud Optimized GeoTIFF (a few MB) and kept in `cache/hrdem/`, so no big download. Where the mosaic has no LiDAR, the standard elevation fills in (shifted to match). With *Quebec LiDAR* also ticked, the Quebec sheets are used on top of it (they also bring the tree heights).
   **Show buildings on map** loads the buildings from OpenStreetMap. Red buildings sit on the track and are removed automatically. Click any building to keep or remove it.
4. **Generate**: shows stats, warnings, and the elevation profile. Use **View in 3D** to preview the model (the *Start line view* button shows the grid from driver height).
   Generating also writes `output/<name>/ac_track/<name>/<name>.kn5`, ready to drive.
5. **Install to AC** copies the track folder into `content/tracks`. The AI line (`ai/fast_lane.ai`) and pit lane (`ai/pit_lane.ai`) are generated along the road centre (target speeds from the corner radii, braking and acceleration limits), so AI cars drive right away. For a real racing line, record one lap with Content Manager's AI line recorder and replace `ai/fast_lane.ai`. (Area mode: no AI line, free roam.)
6. To fine-tune: **Build Blender scene** runs Blender in the background and produces `output/<name>/blender/track.blend` (and `track.fbx`). Edit it in Blender (see below) and **save**.
7. **Build KN5** → *From my Blender scene* converts `track.blend` to `<name>.kn5` (Blender runs in the background). *From track.fbx* or *From another FBX file…* converts an FBX instead. Then **Install to AC** again.

## Area mode (free roam)

Like Map2Lap's areas: instead of one track, every drivable street inside a rectangle or a polygon becomes road (`app/network.py`).

* **Rectangle**: drag it on the map. **Polygon**: click its corners, then click the first one (or double-click) to finish; drag a corner to move it, right-click it to remove it, drag a small **+** on an edge to add one; Esc cancels. At most **10 km²**, and 6 km across. Streets, ground, buildings, trees and street furniture stop at the outline. **Start spot**: click it, then click the street where the cars should start; otherwise they start on the main street nearest the middle, facing into the area (*Reverse direction* turns them round; on a one-way street they always face the way traffic goes). The pit and grid spots line up behind it, two abreast on a street wider than 5.5 m. There are no timing gates (free roam).
* **Big areas** (bounding box over 4 km²): to keep memory, generation time and the .kn5 as for a 2 x 2 km area, the ground texture (0.5 m, or 1 m above 16 km² of box), the photo analysis (vegetation, buildings, driveways: 0.5 m x the scale) and the sharp ground beside the streets (15 m) get coarser; the log says what it used. Street widths and markings are still read on the photo at full resolution, piece by piece (`photo.TiledPhoto`). On the Saguenay test a 9 km² area took about 4 minutes, at most 4 GB of memory, and made a 127 MB .kn5; a 4 km² area at full detail peaks at about 5 GB.
* **Streets**: the OpenStreetMap streets in the rectangle, cut at its edge: motorways to residential streets, living streets, service roads (not private driveways or parking aisles: the photo finds those), raceways. Footways, paths, farm tracks and tunnels are left out. The pieces of a street between two junctions are one street, so each one gets the full track treatment:
  * width measured on the photo and smoothed (inside junctions the width comes from either side; streets the photo can't read take the typical measured width of their class, e.g. Saguenay residential 7.5 m);
  * markings read on the photo; when the photo can't tell, main roads (tertiary and up) get the standard lines and residential streets and service roads none. Lines stop inside crossings and across the mouth of a side street; the through street's centre line runs on through a T junction;
  * heights smoothed over the whole street network at once (*Elevation smoothing*), so streets that meet have exactly the same height there.
* **Junctions**: the asphalt is the union of the streets with 7 m kerb radii at the corners, and a turning circle at the end of a cul-de-sac where OpenStreetMap maps one (`highway=turning_circle`). On a hillside the streets meeting at a junction disagree (one runs up the slope, the other across it): a plane fitted through the junction takes the warp and hands over to each street over 10 m, and the crown fades out there, so there is no ridge or step. On the test junctions the surface changes by at most 1.8 cm per 25 cm (the hill's own 5 % is 1.25 cm).
* Everything else is built as for a track: one ground surface with the streets in it, buildings (doors face the nearest street), driveways, trees, grass, street furniture, CSP files. Shoulders, barriers and tyre walls are route-only.
* **Preview road on map** shows every street (edges green where the photo is clear), the markings and the start spots; right-click a street to set its width.
* **Limits**: streets missing from OpenStreetMap are not built (the photo's paved areas still become drivable patches up to 6000 m²). A bridge or overpass is built as a level crossing (the ground can't pass under a road); the log says how many. A 2 x 2 km area with 0.25 m ground texture makes a .kn5 of about 130 MB; *Ground detail: Medium* makes its ground textures four times smaller.

## rFactor 2 (Mod Mode)

After a track is generated, **Export to rFactor 2** (Result panel) turns the same model into an rF2 *location* and installs it in `<rFactor 2>\ModDev\Locations\<name>` (`app/rf2.py`, `blender/export_rf2_gmt.py`).

* **Set up once** (⚙ Settings → rFactor 2): the rFactor 2 folder (e.g. `D:\SteamLibrary\steamapps\common\rFactor 2`; found by itself in the usual Steam places) and a **Blender 5.0.x** for the exporter: the R25.11 add-on is compiled for Python 3.11, so it loads only in Blender 5.0 (Blender 5.1 and later run Python 3.13 and fail with *bad magic number*). Easiest: unzip the portable `blender-5.0.x-windows-x64.zip` into the maptotrack folder, it is found by itself; or set its `blender.exe` in Settings. Your main Blender (for AC) can stay on any version. Put Studio 397's exporter zip (`io_rfactor2_gmt_50.zip`) in the maptotrack folder; Settings says whether it was found. The add-on is installed into that Blender automatically, nothing to click. If no Blender fits, the export still writes everything else and says which Blender it needs.
* **Drive it**: start `Bin64\rFactor2 Mod Mode.exe` (or *Mod Mode* from Steam), pick the track in the track list, any car. Run Export again after re-generating: the old copy in ModDev is replaced.
* **Speed**: the exporter writes about 20 000 triangles a second per Blender, so the app runs several Blenders at once (up to the number of CPU cores - 1, at most 8) and keeps every `.gmt` in `output\<track>\rf2_gmt_cache`: exporting again without re-generating (or after a change that only touches the track's text files) reuses them and takes a few seconds. The add-on is unpacked once into `cache\rf2_addon` and loaded from there (nothing is installed into your Blender).
* **What is written**: one `.gmt` per mesh of the model (big meshes cut into pieces of at most 21 000 triangles, the GMT limit), the textures as `.dds` (BC1, or 32-bit with mipmaps for cut-outs such as trees and grass), and the text files:
  * `<name>.scn` (instances: collision on roads, ground, walls and kerbs; trees, buildings and grass drawn only; timing gates `xfinish`, `xsector1`, `xsector2` invisible);
  * `<name>.gdb` (name, location, latitude/longitude, time zone), `<name>.tdf` (surfaces: asphalt, gravel, grass, rough, dirt, sand, walls, tyres; the material names carry the code);
  * `<name>.aiw` (a fast line along the centre every ~4 m with the road widths, the three sectors, a pit lane along the road through the pit spots, the grid and pit spots from the AC spawn points, each reached by a 2-waypoint branch) and `<name>.cam` (TV cameras every 250 m).
  * The sky dome is copied from the stock Joesville location (it ships with ModDev; the dome has no texture, rF2's sky shader draws it), so the track doesn't depend on Joesville.
  * Menu pictures named like the stock tracks: `<name>icon.tga` (the track on a red tile), `<name>Thmb.tga` (the track in yellow, white dot at the start) and `<name>_loading.jpg` (the preview).
* **Axes**: maptotrack's local frame is x east, y north, z up; rF2 gets (x, z, y), i.e. Y up and Z north (the exporter's own Blender → rF2 mapping). Car headings use rF2's forward = (−sin yaw, −cos yaw).
* **Limits (first version)**: in area mode the AI line and the timing gates follow only the start street (the gdb says *Free Roam*); on a point-to-point route the AI line is closed back to the start; the pit lane runs on the road itself, beside the racing line, past rF2's own pit boxes (in single file 8 m apart, 30 m after the grid at the right of the road; AC's pit spots are the grid spots, which rF2 doesn't accept as pit boxes), with a short branch into each grid spot and pit box like the AIW editor makes (rF2 crashes on the first physics step without them), no lights, no wet/reflection maps, one material per mesh (L2DIFFUSET0). AI drive but along the centre, not a racing line.
* **Outside Mod Mode (Package for rFactor 2)**: normal rFactor 2 only loads installed components (`.mas` archives in `Installed\Locations\<name>\<version>`, installed from an `.rfcmp`). The button:
  1. exports the location first if the ModDev copy is missing or older than the last generation;
  2. packs it with ModMgr's command line into `<name>.mas` (.tdf), `<name>_gmts.mas`, `<name>_maps.mas` and `<name>_layout.mas` (.gdb, .scn, .aiw, .cam, menu pictures and the collision cache rF2 writes when you drive it in Mod Mode), in `output\<track>\rf2_package\<version>`;
  3. adds the component to MAS2's list (`%APPDATA%\cmpinfo.dat`, a backup of yours is kept as `cmpinfo_before_maptotrack.dat`) and opens **MAS2**. There click the yellow box button (*Mod Packager*) → **Create Single Cmp Package** → **Package** (the track, version, .mas files and output file are already filled in), then close MAS2. Making the `.rfcmp` has no command line, so this one step is yours;
  4. installs the `.rfcmp` (`ModMgr -i`), removing older versions of the same track. Start rFactor 2 normally: the track is in the track list.

  Each package gets the next version (1.00, 1.01 …). The `.rfcmp` in `rf2_package\<version>` is what you give to others (they install it with ModMgr or by double-clicking it). Close MAS2 before clicking the button. To race it online, a server also needs an `.rfmod` (MAS2 → *Create New Mod Package*, with cars), not made by the app.
* **Test**: `python tests/rf2_test.py`; with `MAPTOTRACK_TEST_BLENDER=<blender 5> MAPTOTRACK_TEST_RF2_ADDON=<zip>` it also writes real `.gmt` files and checks their axes.

## How the KN5 is made

* `app/kn5.py` writes the KN5 (version 5) directly: textures (PNG, embedded), materials, one mesh node per object and material, and the `AC_` empties as dummies. Meshes over 65,535 vertices are split; the pieces keep the name prefix (`1ROAD…`), so physics still works.
* **From Blender / FBX**: `blender/export_kn5.py` reads every mesh object that is visible in renders (modifiers applied), splits it per material, takes the first Image Texture of each material as the diffuse texture, and converts Blender's Z-up axes to AC's Y-up. Empties named `AC_…` become spawn and timing points. Objects or materials whose name starts with `__` are left out. You can also run it yourself from Blender's Scripting tab.
* **Shaders**: `ksPerPixel` everywhere, `ksPerPixelAT` (alpha tested) for `trees*`, a little specular on asphalt. To change them, create `kn5_materials.json` next to `start.bat`, e.g. `{"^siding": {"props": {"ksSpecular": 0.2}}, "^my_glass": {"shader": "ksPerPixelAlpha", "blend": 1}}` (regex on the material name).
* **Surfaces**: `data/surfaces.ini` defines ROAD and GRASS (grip, sounds, track limits). WALL is deliberately *not* a surface: AC only treats `1WALL…` meshes as solid walls when WALL isn't defined there (otherwise cars drive through barriers and buildings). Older files written by maptotrack with WALL are replaced automatically; a file you edited yourself is kept.
* **Texture compression**: opaque textures go into the KN5 as DDS (BC1/DXT1) with full mipmaps, power-of-two sizes; transparent ones (trees, grass) stay PNG. A 2048 px ground tile uses 2.7 MB of video memory instead of 22 MB, and the KN5 is about 3× smaller. `"kn5_dds": false` in the project turns it off.
* **Draw distances**: grass 90/200 m, trees 1.2 km, driveways 1 km, buildings 2 km, streets 2.5 km, walls 1.5 km; road and terrain always. AC loads the whole track, so this saves frame time.
* **Trees and grass** use the `ksPerPixelAT_NS` shader (no self-shadow stripes on cards with upward normals), and atlas images have a transparent margin (no thin lines at a distance).
* The old route (`export_fbx.py` → ksEditor) still works if you prefer ksEditor's material editor.

Re-generating a project keeps everything in `output/<name>/`. *Open project…* (top left) restores waypoints, settings and building choices.

## What gets generated

| Object (Blender/FBX name) | Meaning in AC |
|---|---|
| `1ROAD_ASPH_jj_ii` | track and street surface: tiled photo asphalt (`asphalt_tiled`, repeats every 2.5 m) - sharp up close (physics ROAD) |
| `1GRASS_SHLD_jj_ii` | the shoulder: tiled photo gravel (`gravel_tiled`) (physics GRASS) |
| `1GRASS_NEAR_<SURFACE>_…`, `1ROAD_NEAR_…` | ground within *Sharp ground beside track* m of the track: tiled photo materials `near_lawn`, `near_wild`, `near_dry`, `near_forest`, `near_sand`, `near_soil`, `near_gravel`, `near_paved` |
| `1ROAD_GND_jj_ii` | driveways / parking lots on the ground texture (physics ROAD) |
| `1GRASS_GND_jj_ii` | all other ground triangles (GRASS). Same surface, same texture as `1ROAD_GND` - no seam |
| `ROAD_LINES`, `ROAD_LINES_Y` | white / yellow road markings (as painted, see *Road markings*), 1.5 cm above the asphalt (visual only, never touched by physics) |
| `1WALL_000…` | concrete barriers (collision) |
| `1WALL_BLD_…` / `BLD_ROOF_…` | building walls (solid if enabled) / roofs (visual) |
| `BLD_STEPS` | concrete steps in front of house doors (visual) |
| `TREES_…` | trees, bushes and hedges (visual only, alpha-tested `trees` material) |
| `VEG_GRASS_…`, `VEG_TALLGRASS_…` | 3D grass tufts (`veg_grass_short`, hidden when CSP Grass FX runs) and tall grass (`veg_grass`) - visual only, drawn up to 90 / 200 m |
| `1WALL_PROPS_…` | posts and poles of signs, traffic lights, street lights and utility poles, guard rails (solid) |
| `PROPS_…` | sign panels, signal heads, mast arms, lamp heads, cross arms (visual only, drawn up to 900 m) |
| `PROPS_LENS_…` | street-lamp lenses (material `props_lit`: lit at night with CSP) |
| `ROAD_CROSSWALKS_…` | painted pedestrian crossings (`road_paint`, visual) |
| `1WALL_PYLON_…` / `PYLONS_…` | power-line towers and substations: legs, fences, transformers (solid) / the rest (visual), drawn up to 4 km |
| `WIRES_HV_…`, `WIRES_…` | power-line conductors / street-pole wires (visual, drawn up to 2.5 km / 500 m) |
| terrain materials `ground_jj_ii` | one aerial-photo / land-cover texture per 512 m terrain tile |
| `AC_START_n`, `AC_PIT_n`, `AC_HOTLAP_START_0` | spawn points (Y up, Z = driving direction) |
| `AC_TIME_n_L/R` | timing gates (circuits: 0 = start/finish, 1–2 = sectors) |
| `AC_AB_START_L/R`, `AC_AB_FINISH_L/R` | start/finish gates for point-to-point tracks |

Every mesh stays under the KN5 limit of 65k vertices. The track folder also gets `<name>.kn5`, `ui/ui_track.json`, `ui/outline.png`, `ui/preview.png`, `map.png`, `data/map.ini` and `data/surfaces.ini`.

## How the road is joined to the ground

The road is integrated into the ground, the way Map2Lap does it. The track is not a strip laid on top of the terrain; the terrain *is* the road:

1. One triangulated surface (Delaunay TIN) is built from three kinds of points: the road cross-section (7 points across, every 2 m, crown included), a fine 2 m grid within 14 m of the track and along streets, and the coarser terrain grid further out. Road points take the smoothed road height, everything else the terrain height, so the ground meets the asphalt edge exactly.
2. Each triangle is named by where its centre falls: inside the track, a street or a driveway → `1ROAD_GND_…` (ROAD physics); anywhere else → `1GRASS_GND_…` (GRASS). Both use the same per-tile ground texture.
3. Asphalt (track, streets, gravel shoulder band) and driveways are painted into that texture with a little grain and a darker rim; markings are a thin separate mesh, `ROAD_LINES` (see *Road markings*).

Result: no seams, no kerbs, no z-fighting. On the Saguenay test circuit the measured step at the road edge is 0.0 cm (90th percentile 0.1 cm). The road's look depends on the ground texture resolution (*Ground detail*): use High (0.25 m/px) for sharp edges.

* *Ground: Plain* gives mown-grass green ground with the asphalt painted in.
* *Terrain* off builds only a strip of ground along the road (shoulder or barriers + 3 m) and the streets, for when you add your own scenery around it.

## Ground materials (ambientCG photos)

**What you see from the car** (the first in-game test showed why: a 25 cm-per-pixel ground texture is a blur at eye height):

* **Track and streets** have their own material: a photo asphalt (Asphalt031, stains removed so the repeat doesn't show) repeated every 2.5 m, about 2.5 mm per pixel. Street edges are vertices of the ground surface, so the material ends exactly at the kerb line. A street (or mapped raceway) that the track itself follows is left out where it runs along the track, so its own centreline and width don't poke out past the track edges. Markings stay a separate mesh.
* **Shoulders** (*Shoulder* width) get a tiled photo gravel the same way.
* **The ground beside the track** (*Sharp ground beside track*, default 30 m) is split by surface - lawn, wild grass, dry grass, forest floor, sand, soil, gravel, driveways - and each surface gets a real photo (Grass005, Ground037, Gravel043, Asphalt031) repeated every 2-3 m, tinted to that surface's colour in the ground texture so it joins the ground further away. The surface is the land-cover colour nearest to the ground texture, read every 0.5 m (ignoring the asphalt painted under streets), specks under 3 m2 merged, shapes rounded; driveways come only from the driveways found on the photo. The outlines of these regions, the track edges, the street edges and the shoulder edge (points every 0.8 m, also around the outside of tight hairpins) are vertices of the ground, so materials meet on smooth lines instead of along 1-2 m triangles. Ground points are every 1 m within 15 m of the track. The textures are shared, so this costs vertices rather than texture memory. 0 turns it off.

CC0 photo textures from [ambientCG](https://ambientcg.com) (Asphalt031, Grass005, Gravel043, Ground037, plus Plaster001 / Concrete034 for houses) are downloaded once into `cache/textures` and used two ways:

* **Baked** into the ground tiles (land-cover colours and *Plain*; an aerial-photo ground already has its own texture). At 25 cm per pixel this gives natural patches and wear rather than visible blades or stones: those are smaller than a pixel.
* **Close-up detail in Assetto Corsa** (*Close-up ground detail (experimental)*, off by default - in the first test it turned the ground grey): the ground uses AC's `ksMultilayer` terrain shader. Each tile gets a 0.5 m mask of where asphalt / grass / gravel / dirt-sand are (read from the painted tile), and four photo details (grey, so the ground keeps its colours) repeat every 2-3 m near the car: asphalt aggregate on the roads, grass on lawns and fields, stones on shoulders and lots, soil on sand and dirt. Ground tiles are square so the detail has the same scale both ways. The 3D preview in the app shows the baked tile only.
  * Settings written: `magicMult` 2, `multR/G/B/A` = repeats per tile (tile size / 3, 2, 2, 3 m). If the ground looks too dark, too light or wrong in game, untick the option and regenerate, or override the `^ground_` materials in `kn5_materials.json` (e.g. another `magicMult`).
  * Going through Blender (*Build KN5* from track.blend / FBX) gives plain `ksPerPixel` ground without the close-up layer.

Power-line towers and poles are too thin (5-15 cm beams) for a photo texture to show; they keep flat colours.

## Street furniture and power lines

Under *Street furniture* (Scenery card), from OpenStreetMap, the way Map2Lap does it. All of it keeps clear of the track and its barriers, of the streets' carriageways and of buildings.

* **Power lines**: OSM `power=line` / `minor_line` (underground ones are left out). A tower at each mapped tower/pole node, or at each corner of the line: steel lattice pylons (33 m) for transmission lines (69 kV and up, or no voltage given), wooden poles for lower-voltage lines. The wires hang between them with a sag. A tower that would stand on the track or a street is skipped and the wires span over it.
* **Traffic lights & signs**:
  * *Traffic lights* where OSM maps a `highway=traffic_signals` node at a junction (or mid-block): one per way in, at the kerb on the drivers' side, facing the traffic coming in. North American style: a post head plus a head on a mast arm over the lane. The heads on the major road show green, the others red.
  * *Stop / give-way signs*: mapped `highway=stop` / `give_way` nodes; where none are mapped, the usual rules: stop on minor roads meeting a major road, on the stem of a minor T, all-way stops at half of the minor 4-way junctions (always the same ones), give way when entering a roundabout.
  * *Speed-limit signs* at the start of a road whose OSM `maxspeed` differs from a road the traffic can come in from, 25 m in, facing the drivers.
  * *Sign style*: Auto picks Québec (ARRÊT, number in a red ring), Canada (MAXIMUM), USA (SPEED LIMIT, mph) or Europe (round signs) from the location.
* **Street lights & utility poles**: mapped `highway=street_lamp` nodes. In the Americas: wooden utility poles every 40 m on one side of town streets, with their wires and a cobra-head lamp on every other pole (not where OSM maps the low-voltage lines itself). In Europe: lamp posts every 30 m along lit or built-up streets. The lamp lenses are their own mesh (`PROPS_LENS`, material `props_lit`) so CSP can light them at night (below).
* **Substations** (with *Power lines*): OSM `power=substation` areas (not underground / indoor / minor distribution). A 2.4 m chain-link fence along the outline (see-through, collides), transformers (tank, radiators, conservator, bushings) at the mapped `power=transformer` nodes or spread along the yard, and for high-voltage yards (69 kV and up) steel gantries with insulators and breakers. Nothing else (poles, signs) is put inside.
* **Crosswalks** (*Crosswalks*): OSM `highway=crossing` nodes with markings (`crossing=marked / uncontrolled / zebra / traffic_signals`, `crossing:markings=*`; unmarked and bare crossings are skipped) are painted on the road they cross, the track or a street, across its whole width: zebra stripes (0.5 m, 3 m long), or two 0.3 m lines for `crossing:markings=lines` and for signalled crossings in North America. Mesh `ROAD_CROSSWALKS` (material `road_paint`), 2.5 cm above the ground surface.
* **Guard rails** (*Guard rails*): OSM `barrier=guard_rail` ways become W-beam rails (0.5–0.82 m) on posts every 2 m, ends turned down into the ground, on the road side of the posts. They collide (`1WALL_PROPS`). A rail that OSM draws on the track's asphalt is moved 0.8 m beyond its edge; pieces on a street or in a building are left out.

The furniture is a separate small download (`cache/osm/f2_*.json`; `fu_*.json` were the files from before crossings, guard rails and substations), so turning it on doesn't re-download the rest. If it fails, the rule-based signs and poles are still built.

## Fine-tuning options

* **In the app**: adjust waypoints, segment modes, smoothing, width, start offset, and which buildings to keep, then regenerate. It takes a few seconds.
* **Blender**: everything is organised in collections (ROAD, TERRAIN, BUILDINGS, WALLS, AC_OBJECTS). `1ROAD_GND_…` and `ROAD_LINES` are in ROAD, `1GRASS_GND_…` in TERRAIN.
  * The [AC Tools](https://extensions.blender.org/add-ons/ac-tools/) add-on (Blender 4.2+) understands this naming. It can validate the scene and manage surfaces and the AC_ objects.
  * [BlenderGIS](https://github.com/domlysz/BlenderGIS) or Blosm can add satellite imagery, more OSM detail or Google 3D tiles on top.
  * Replace the placeholder textures in `blender/texture/`. Keep the file names or re-link them in Blender.
* **Custom Shaders Patch** (CSP): the app writes `extension/ext_config.ini` (ignored without CSP):
  * Grass FX and CSP trees (see Vegetation);
  * *CSP night lights*: one light under every street lamp (`[LIGHT_SERIES_0]`, spot down, warm white, 26 m range; the 1500 lamps nearest the track at most) and the lamp lenses glowing (`[MATERIAL_ADJUSTMENT_0]`, `ksEmissive` on `props_lit`), both on CSP's built-in `NIGHT_SMOOTH` condition (sun below 88°);
  * *CSP rain* (RainFX): puddles on the asphalt, soaking (darker) asphalt, gravel, soil and sand, rough (no shine) grass, ground, trees and roofs, raindrops on the metal props, and the road paint as slippery `LINES_MATERIALS`.
  Delete the first line of the file to keep your own edits when you generate again.

## Speed (parallel generation)

* **Downloads all start at once**: elevation tiles, LiDAR sheets (MNT and MHC together), the aerial photo pieces, OpenStreetMap and Overture all download in parallel while the track is being prepared. Elevation tiles and photo pieces are themselves fetched several at a time.
* **Heavy steps use every CPU core**: ground textures (one 512 m tile per core), driveway painting, building detection (each candidate building, alignment, shadow test), PNG/JPEG encoding for the texture folder, the preview and the KN5 (each texture is now encoded once instead of twice). numpy, scipy, Pillow and GDAL release Python's lock while they compute, so threads run truly in parallel without copying data between processes.
* The result is identical whatever the number of cores (checked by `tests/parallel_test.py`).
* `MAPTOTRACK_WORKERS=1` (environment variable) runs everything one step at a time, e.g. to debug; another number caps the cores used.
* Newer scipy releases the lock in more of its functions: `pip install -U scipy` inside `.venv` can make building detection scale better.

## Troubleshooting

* **Build KN5 says Blender not found**: set the path to `blender.exe` in Settings (⚙).
* **A building or object is missing in game**: check it is visible in renders in Blender (camera icon) and its name doesn't start with `__`.
* **Track 100× too big or small in ksEditor** (old route only): change `APPLY_SCALE` at the top of `export_fbx.py` and re-export.
* **Car or kart stuck at the start / looks sunk into the ground**: regenerate with the current version. The road and the ground are one continuous surface, so there is no edge to get stuck on, and spawn and timing points are placed on the asphalt that was actually built (the photo-measured road can be off-centre from your line), 30 cm above its surface.
* **Cars spawn backwards**: set `FLIP_SPAWN_DIRECTION = True` in `build_scene.py` and rebuild.
* **Folded road in tight corners**: raise *Corner smoothing* or reduce *Width*. The app warns you when the corner radius is too small.
* **Bumpy road**: raise *Elevation smoothing* (10–15 m for a kart track, 25 m for roads). The road height is the median of five LiDAR/DEM samples across the road every 2 m, and bumps shorter than ~10 m are always smoothed out, but at 0–5 m the remaining terrain detail still feels like hits at kart speed. The app warns when the profile has bumps over 3 cm or slope changes over 3% within 2 m.
* **"502 Bad Gateway" / "server busy" from overpass-api.de**: the free OpenStreetMap server is overloaded, not a problem on your side. The app then tries overpass.private.coffee and maps.mail.ru, and the whole list once more after 20 s. Downloads are saved in `cache/osm` and reused for any route inside the same area, so after one success, editing the route doesn't download again.
* **Data downloads fail**: the public servers are rate-limited. Try again later or point to your own servers with these environment variables: `MAPTOTRACK_OSRM_URL`, `MAPTOTRACK_OVERPASS_URLS`, `MAPTOTRACK_TERRARIUM_URL`.
* **Try it without internet**: `set MAPTOTRACK_OFFLINE=1` before `start.bat` switches to synthetic demo data.

## Data sources & licences

* Roads, buildings and streets: © OpenStreetMap contributors, ODbL. The attribution is written into `ui_track.json`. Keep it if you publish the track.
* Aerial photos (Quebec): Orthophotographies Saguenay–Lac-Saint-Jean 2025, Gouvernement du Québec / MRC du Fjord-du-Saguenay, CC BY 4.0, via Données Québec. The credit is added to `ui_track.json` automatically. To add another region, append a WMS entry to `IMAGERY_SOURCES` in `app/imagery.py` (most Données Québec regions publish one).
* Land cover: ESA WorldCover 10 m (via Overture Maps), CC BY 4.0.
* Building and ground material textures: [ambientCG](https://ambientcg.com), CC0.
* LiDAR: Lidar – Modèles numériques (terrain, canopée), Gouvernement du Québec (MRNF), CC BY 4.0, via Données Québec.
* HRDEM (Canada): High Resolution Digital Elevation Model mosaic, Natural Resources Canada, Open Government Licence – Canada (credit added to `ui_track.json`).
* Extra buildings where OSM has none: [Overture Maps](https://overturemaps.org) building footprints (OSM plus Microsoft/Google AI-detected footprints), ODbL. Untick *Fill missing buildings* to use OSM only.
* Routing: OSRM public demo server and FOSSGIS routed-foot (fair use).
* Elevation: AWS Terrain Tiles (Terrarium; SRTM, NED, Copernicus and others).
* Google Maps data (buildings, imagery) is **not** used, because its terms don't allow extracting it. Google links and My Maps exports are only read for your own waypoints.

## Project layout

```
app/        server.py (FastAPI), routing.py, osm.py, elevation.py, track.py, meshes.py, ground.py (integrated road surface), props.py (street furniture),
            houses.py (clean house footprints, doors, garage doors), network.py (area mode: street network),
            textures.py + materials.py + groundmat.py (house facades, ambientCG photo materials, ground detail),
            ac_export.py, kn5.py, rf2.py (rFactor 2 location), parallel.py, pipeline.py
web/        map UI (Leaflet) + 3D preview (three.js), libraries vendored in web/vendor
blender/    build_scene.py, export_fbx.py (copied into each output), export_rf2_gmt.py (rF2 .gmt files)
output/     one folder per track
cache/      downloaded OSM + elevation tiles
tests/      smoke_test.py, kn5_test.py, props_test.py, driveways_test.py, csp_test.py, markings_test.py, tyres_test.py ... (MAPTOTRACK_OFFLINE=1 python tests/smoke_test.py)
csp_trees/  your CSP .bin tree models (not in the repository; see README.txt there)
```

`output/`, `cache/`, `settings.json`, `.venv/` and the rFactor 2 exporter zip are local only (see `.gitignore`).

## Running the tests

The tests need no network: with `MAPTOTRACK_OFFLINE=1` the app uses synthetic map data. Each test is a plain script.

```
# Windows (PowerShell), inside .venv
$env:MAPTOTRACK_OFFLINE = "1"; Get-ChildItem tests\*_test.py | ForEach-Object { python $_.FullName }
# Linux / macOS
for t in tests/*_test.py; do MAPTOTRACK_OFFLINE=1 python "$t" || break; done
```

GitHub Actions runs them on every push (`.github/workflows/tests.yml`). `rf2_test.py` also runs the real rFactor 2 exporter when `MAPTOTRACK_TEST_BLENDER` (a Blender 5.0.x) and `MAPTOTRACK_TEST_RF2_ADDON` (the exporter zip) are set.

## How the road width is measured

1. The photo is sampled on a strip that follows the route: one row every 2 m, one sample every 10 cm across, up to 15 m each side.
2. The colour of the asphalt (CIE Lab + fine texture) is learned from the middle of the road over a sliding 80 m window, so old, patched or new asphalt all work. Faint white edge lines count as road.
3. The middle of the asphalt is found first, which corrects OpenStreetMap lines that sit off-centre (common in corners).
4. Each edge is traced along the whole track with dynamic programming. It looks for the innermost clear asphalt-to-ground boundary and moves smoothly from one station to the next, so parked cars, shadows, side streets and driveways are bridged instead of making spikes.
5. Each edge is then placed at the exact 50 % point of the colour change (sub-pixel), then lightly smoothed.
6. **Width smoothing** (Road card, default 30 m; 0 = raw measurement): the width and the road's centre are averaged along the track over that length. A running median first removes short spikes (a driveway or side street read as road, a car or shadow read as verge), then a Gaussian weighted by the edge confidence: stretches where the edges are clear on the photo count most, and faint ones take their width from their neighbours. Real changes of width longer than the smoothing length are kept. Manual widths (below) are applied afterwards and stay exact.

On a synthetic photo with a 7 m and an 11 m section, the app measures 6.9 m and 10.9 m (also with 30 m smoothing). On the Saguenay test circuit the raw measurement gives 6.6–8.8 m (median 7.6 m); with 30 m smoothing 7.4–8.1 m, with 60 m 7.4–7.9 m.

**After a road preview (or generating), check and fine-tune on the map:**
* Road edges are drawn **green** where the boundary is clearly visible on the photo and **orange** where it's faint (e.g. light asphalt next to light sand). The *Road width along the track* chart shades the faint stretches.
* **Click the track** (the white dashed line) to see the built width at that spot and **set the width by hand** over 20 m to the whole track. Manual widths are listed in the Road card, saved with the project, and keep the measured road centre. With *Live* on, the preview updates right away; otherwise preview or generate again.

## Road markings

*Road markings* (Road card): **From photo (as painted)** (default), **Standard** or **None**.

From photo, the app looks for the lines really painted on the track, so it gets them where they are, or none at all on a road without paint (many residential streets in Québec have none: the Soleil-Levant circuit gets no lines, Route Mathias next to it gets two white edge lines and a yellow centre line). `app/markings.py`:

1. The photo is resampled along the track: a row every 0.25 m, a column every 5 cm across.
2. A painted line (10-15 cm wide, about 0.3 m on a 20 cm photo) is a narrow ridge lighter than the asphalt on *both* sides, so the asphalt / gravel edge doesn't count. Yellow paint on light asphalt is only half as bright, so it is found from a weaker ridge that is also yellower (the photo's colour is blurrier than its lightness).
3. Each line is followed row by row, like tracking objects in a video, so it stays on the paint even where the mapped centreline wanders across the road. Pieces of the same line split by worn paint are joined using a parallel line as a guide. Tree shadows, leaves and cars are 'not seen' and the line carries on through them.
4. Style along the line: solid, or dashed when short runs repeat at one period, in which case the dash length, period and position are taken from the photo, so the dashes are where they really are. Colour: white or yellow. Position: as measured, kept on the asphalt.
5. If less than 40 % of the road can be seen on the photo, or there is no photo, the standard layout is used and a warning says so.

*Preview road on map* draws the lines found (white / yellow, dash by dash), and the result card shows how many there are. Standard is the old layout: white edge lines 12-27 cm inside the asphalt and a dashed white centre line.

## Tyre walls

*Tyre walls* (Road card): **Auto (race tracks)** (default), **From photo**, **None**.

On a 20 cm photo a tyre wall is a dark, beaded band 0.6-1 m wide beside the track. `app/tyres.py` looks for it 0-5 m outside each asphalt edge (less on the inside of tight corners): a band darker than its surroundings, and on grass also greyer (a black tyre on grass is hardly darker on a blurred photo, but it isn't green), dark on both sides (not the edge of a field), and not green like a hedge. The band is followed along the track like the road markings; gaps under 4 m are bridged, walls under 6 m dropped. The wall stays where the photo shows it (centre of the tyres at least 0.35 m off the asphalt).

Each wall becomes stacks of 3 (sometimes 4) real tyres every 0.58 m: each tyre is a car tyre's cross-section turned around its axis (rounded shoulders, tread with blocks and grooves, sidewalls, the hole in the middle; 0.60 m across, 0.20 m wide), a little different in size, turned and tilted, the sidewalls squashed a little under the weight. These detailed tyres (`TYRES_…`, about 200 vertices each) are drawn up to 70 m; a simple 8-sided stack hidden inside them (`1WALL_TYRES`) is always drawn: it is the collision (karts bounce off it) and what is seen from further away. Material `tyres`. No tree stands in a tyre wall. The tyres' texture is drawn by the app (tread blocks and grooves, sidewalls, beads; a far-view stack; the dark inside) with the grain of a real rubber photo: TextureCan *Clean Car Tire Rubber* (Plastic 0022, CC0), downloaded once into `cache/textures/TextureCan_Plastic0022_1K_Color.jpg`. Any rubber photo saved under that name is used instead; without one, a procedural grain.

*Auto* finds tyre walls only when OpenStreetMap maps the track as a race track (`highway=raceway` along at least 40 % of it, as SPK-kart): on public roads, tree shadows, ditches and fences look like tyres too (on the Soleil-Levant circuit the detector would find 360 m of 'walls'). On SPK-kart it finds about 640 m. *Preview road on map* draws them (dark dashed lines).

## How buildings are found

* **With Quebec LiDAR on** (best): the canopy-height model includes buildings. Smooth, roof-like surfaces 2.2 m or more tall become footprints with their real eave and ridge heights (15th and 95th percentiles of the roof). Tree crowns are rejected because they are rough, round or green on the photo. The LiDAR survey is older than the 2025 photo, so the photo pass below also adds houses built since.
* **On the photo**: roof colours are learned from buildings that are already known (mapped buildings, LiDAR finds, or, if nothing is mapped, Overture footprints used only for learning). Every roof-coloured blob must be compact and nearly rectangular, and must cast a shadow in the same direction as the known buildings. That rejects driveways and parking lots with roof-like colours. Merged blobs (house + driveway, two neighbours) are split before the test.
* Outlines that are almost rectangular become clean rotated rectangles of the same area.
* Without LiDAR, heights come from footprint size: sheds, bungalows up to 300 m² (incl. attached garage), two storeys above that.
* Houses that appeared after the LiDAR survey ignore LiDAR heights that look like the trees that stood there before (eave above 7.5 m).

## How houses are built

* **Clean footprints** (`app/houses.py`): outlines traced from the photo or LiDAR have cut corners, notches and wobbly edges. Each one is redrawn square to the house's own direction (from its longest walls): it is rasterised at 25 cm, small bumps and notches under ~1 m are closed, then:
  * a house (under 500 m²) that is close to a rectangle becomes one rectangle with the same area;
  * an L, T or U house becomes a main block plus up to two wings (each at least 2.8 m wide), when they cover 80% of it; otherwise it is still one rectangle, never an odd polygon;
  * only big buildings (over 500 m²) that aren't rectangles, or anything over 1,500 m², keep their outline (with square corners and straight slanted walls) and a flat roof.
* Houses get a 35 cm roof overhang and a shingle roof at a 1:2 pitch. Long houses get a gable roof along the long side, squarish ones a hip roof; wings get a lower gable roof.
* **Doors**: every house has a front door (0.95 × 2.05 m panelled door with three lights, coloured like the shutters, or dark wood on brick) in the middle bay of the wall that faces the nearest street or the track, with two concrete steps up to the ground floor. Bays hidden behind a wing are skipped.
* **Garage doors**: where a driveway found on the photo meets a house wall (within 1.2 m), the bay where it meets gets a 2.6 × 2.15 m sectional garage door (white or almond) that comes down to the driveway, with no foundation under it. On the Saguenay circuit: 110 front doors and 11 garage doors.
* **Walls**: 13 materials - vinyl lap siding in 8 colours (white, cream, beige, grey, blue, sage, yellow, charcoal), brick (red, brown, buff) and stucco (white, sand), weighted like a Québec suburb (mostly siding, about a quarter brick). Each house's material, and whether it has shutters (35%), is picked from its id, so it stays the same between generations.
* **Windows fit the walls**: every wall gets a whole number of window bays (one per ~3.4 m, stretched to fit), so a window is never cut by a corner; walls under 2 m and gable ends are plain. One row of windows per storey, counted down from the eave. Each window has a white casing, four panes with a sky reflection, a sill with its shadow, and optional louvred shutters in a colour that goes with the wall.
* **Foundation**: 30 cm of concrete shows under the siding (more on a slope).
* **Photo materials**: the fine structure of the walls and roofs (plaster, concrete pores, asphalt-shingle granules) comes from CC0 photo textures by [ambientCG](https://ambientcg.com) (Plaster001, Concrete034, Asphalt031), downloaded once into `cache/textures` (about 10 MB). Without them a procedural grain is used. Patterns are periodic per bay and per storey (16 lap boards and 36 brick courses per storey, 15 bricks per bay), so walls have no seams.
* **Porches and door canopies**: 40% of houses (picked from the id) get a covered porch at the front door: a deck at floor height, up to two bays wide and 1.8 m deep (never over a garage door), a shed roof on white posts that stays under the house's roof overhang, the steps in front of the deck. The others get a small canopy (1.9 × 1.0 m shed roof with a white fascia and soffit) over the door.
* **Building types** from OpenStreetMap `building=*` (and `shop` / `office` / some `amenity` tags); untagged buildings by size (under 30 m²: a shed / garage, over 900 m²: a shop block):
  * **house** (`house`, `detached`, `bungalow`, …): as above;
  * **garage** (`garage`, `garages`, `carport`, `shed`): 2.6 m walls without windows, a low gable roof, sectional garage doors on the wall a driveway meets, or else the wall facing the street (one door, two on a garage over 5.5 m deep, every bay for `garages`);
  * **commercial** (`retail`, `commercial`, `office`, `supermarket`, `school`, `church`, … or any building with a `shop` / `office` tag): brick or stucco, a 4 m ground floor with shop fronts (big windows, mullions, a sign band) and a door on the walls facing a street within 60 m, office windows elsewhere and on the upper floors (3.2 m each; from `building:levels`, else one storey, two on big footprints), a flat roof behind a 0.9 m parapet with a cornice (cap, dentils, mouldings);
  * **block** (`apartments` with 3+ levels or over 400 m², and big irregular houses): house windows on every storey (2.8 m), a door on the street, flat roof, parapet and cornice;
  * **industrial** (`industrial`, `warehouse`, `manufacture`, …): plain walls 6 m high (8 m over 3,000 m²), one or two roller doors on the street wall, flat roof.
  The facade atlas has extra rows for these (shop front, office storey, cornice, trim). The log lists how many of each were built.
* Found buildings are drawn in purple on the map after generating. Buildings that would sit on the track are removed as usual.

On the Saguenay test circuit with nothing mapped in OpenStreetMap, LiDAR + photo found 111 buildings (62 from LiDAR, 49 houses built after the survey from the photo), matching nearly every house on the photo.

## Vegetation

The ground is no longer just a texture: trees, bushes, hedges and grass stand on it in 3D.

* **Canopy (trees and bushes)**: with Quebec LiDAR on, anything taller than 1.5 m in the canopy-height model counts, but only where the 2025 photo still looks like a tree there. That drops trees cut since the survey, new houses, and power lines over the road. Each LiDAR tree top becomes one tree with its measured height; crowns are kept inside the canopy patch seen on the photo. Without LiDAR, the photo alone decides. A small model compares how dark, blue and textured each spot is against the scene's typical vegetation, so it adapts to the season and to the photo's colours. Checked against LiDAR on the Saguenay circuit, it finds 74% of tree pixels and is right on 90% of the open ground.
* **Hedges**: long, narrow canopy strips (under 3.5 m wide, at least 6 m long, under 5 m tall) become a dense row of bushes along their centre line.
* **Evergreens vs broadleaf**: darker, bluer crowns are evergreens.
* **Tree models**: three vertical leaf cards at 60° plus a horizontal crown card, so trees look full from every angle and from below.
* **3D grass**: mowed lawns (smooth grass on the photo) get short tufts. Wild, dry or uneven grass, ditches and big meadows or hay fields away from houses get tall grass clumps, green or straw-coloured to match the photo. Grass is densest within ~12–20 m of the track edge and thins out to nothing at 70 m (lawn) and 140 m (tall grass). The amount is capped by *Grass density* and the limits `max_grass` / `max_tall_grass` (30,000 / 25,000 clumps). It is kept off the road, the shoulders, streets, driveways and buildings.
* In Assetto Corsa, grass chunks have a draw distance (90 m short grass, 200 m tall grass) so far-away grass costs nothing.
* **CSP Grass FX** (*CSP Grass FX (3D grass in game)*, on by default): with Custom Shaders Patch, Assetto Corsa grows its own real 3D grass - thousands of blades that sway in the wind and take shadows - instead of the flat tufts. The app writes `extension/ext_config.ini` in the track folder (`app/csp.py`, CSP docs: https://cup.acstuff.club/docs/csp/tracks/grass-fx):
  * lawns beside the track (`near_lawn`): short, tidy grass, mown every 7 days (configuration A);
  * wild and dry grass (`near_wild`, `near_dry`): tall, untidy, uncut grass (configuration B);
  * the ground further out (`ground_jj_ii` tiles): grass only where the texture is green;
  * none on the track, road paint, shoulders, driveways, sand, gravel or soil (grass fades out at sand / gravel edges).
  The app's own short tufts (`veg_grass_short`) are hidden while Grass FX runs; the tall grass clumps (`veg_grass`) stay. Without CSP the file is ignored and the tufts show as before. Grass FX is not shown in the app's 3D preview. To tune it by hand, delete the first line of the file (the app then keeps your version when you generate again).
* **CSP trees** (*CSP trees (your .bin models)*, on by default; does something only once you add models): Custom Shaders Patch has its own tree system - real 3D trees with detail levels, flat stand-ins far away, size / colour variation and seasons (https://cup.acstuff.club/docs/csp/tracks/trees). CSP ships no trees, so:
  * put `.bin` tree models in the `csp_trees` folder next to the app (made with Content Manager: Content > Tools > Creator tools > tree converter, from an FBX; or from a track / tree pack whose licence lets you reuse them). `csp_trees/README.txt` explains it;
  * the kind comes from the file name (pine / spruce / fir ... conifer; birch / maple / oak ... broadleaf; bush / shrub / hedge), or from `csp_trees/trees.json`, which can also give each model's own height so the trees get their real size: `{"my_pine.bin": {"kind": "conifer", "height": 16}}` (without it: 15 m conifers, 12 m broadleaf, 2.5 m bushes);
  * the app copies the models used into the track's `extension/trees/`, writes `trees.txt` (one line per tree: the model of its kind, a random turn, sized to the tree's measured height and crown) and adds `[TREES]` to `extension/ext_config.ini`, snapping them to the ground materials. Kinds without a model borrow one (bushes: a broadleaf model, small);
  * while CSP draws them, the app's own card trees are hidden (`[SHADER_REPLACEMENT_…] MATERIALS = trees`, `LOD_IN = 100000`). Without CSP the track keeps the app's trees. The 3D preview shows the app's trees.

## Photo reading: limits

The photo is read by colour and texture (no AI model). Things that can fool it: light asphalt next to light sand or gravel (faint edges, shown orange), trees overhanging the road, long rows of parked cars, and spring photos where deciduous trees have no leaves yet (fewer trees). Switch any part off if it misbehaves. Tests: `python tests/photo_scene_test.py`, `python tests/lidar_mock_test.py`.

## Roadmap (phase 2)

Kerbs at corners, a racing line (instead of the centre line) for the AI, a separate pit lane road, rF2 night lights and rain, and DDS textures in the KN5 (smaller files).

## License

maptotrack is released under the [MIT License](LICENSE). The bundled Leaflet and three.js keep their own licences, and the tracks it makes contain data with their own terms (OpenStreetMap ODbL and others): see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and **Data sources & licences** above.
