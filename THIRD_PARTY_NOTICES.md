# Third-party notices

## Bundled in this repository

| Component | Version | Licence | Where |
|---|---|---|---|
| [Leaflet](https://leafletjs.com) | 1.9.4 | BSD 2-Clause | `web/vendor/leaflet/` (licence: `web/vendor/leaflet/LICENSE`) |
| [three.js](https://threejs.org) (core, OrbitControls, GLTFLoader, utils) | r160 | MIT | `web/vendor/three/` (licence: `web/vendor/three/LICENSE`) |

## Not bundled: get them yourself

* **Studio 397's rFactor 2 GMT exporter** for Blender (`io_rfactor2_gmt*.zip`): needed only for the rFactor 2 export. Put the zip from Studio 397 in the maptotrack folder. It is not redistributed here.
* **CSP tree models** (`.bin`) in `csp_trees/`: make your own or use a pack whose licence allows it (see `csp_trees/README.txt`).
* **Blender**, **Assetto Corsa**, **Custom Shaders Patch**, **rFactor 2**: installed separately.

## Data the app downloads at run time

The generated tracks contain data from these sources. Keep their credits (the app writes them into each track's `ui_track.json`) if you share a track.

* © OpenStreetMap contributors, ODbL: roads, buildings, street furniture.
* Overture Maps Foundation, ODbL / CDLA: building footprints, land cover (ESA WorldCover, CC BY 4.0).
* AWS Terrain Tiles (Terrarium): SRTM, NED, Copernicus DEM and others.
* Natural Resources Canada HRDEM, Open Government Licence – Canada.
* Gouvernement du Québec (MRNF / Données Québec), CC BY 4.0: LiDAR terrain and canopy models, orthophotos.
* ambientCG, CC0: ground and building material photos. TextureCan: rubber texture.
* Routing: OSRM public demo server and FOSSGIS routed-foot (fair use, not for heavy traffic).
