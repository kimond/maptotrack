Put CSP tree models (.bin) here - maptotrack places them where the real trees are.

* Make them with Content Manager: Content > Tools > Creator tools > tree converter (from an FBX with up to
  three LODs), or use trees from a track / tree pack whose licence lets you reuse them.
* The kind comes from the file name: pine / spruce / fir ... = conifer; birch / maple / oak ... = broadleaf;
  bush / shrub / hedge = bush. Several models of a kind are mixed.
* Optional trees.json gives the kind and the model's own height (m), so the trees get their real size:
    {"my_pine.bin": {"kind": "conifer", "height": 16}, "birch_a.bin": {"kind": "broadleaf", "height": 11}}
* Without CSP the track keeps maptotrack's own trees.
