# PointCloudPlacer

- Each box can be sampled from a gaussian distribution of points
- Each box's movement is averaged over the sample points for that box
- Try to use Ranked List Loss
- Or try to use triplet loss on only the support vectors
- try to impose arc face loss on the box movement
- How do we encode shape? How do we encode box size?