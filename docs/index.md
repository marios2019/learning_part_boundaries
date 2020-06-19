<h1 align = "center">Learning Part Boundaries from 3D Point Clouds</h1>
<p align = "center">
    Marios Loizou &emsp;
    <a href="http://geometry.cs.ucl.ac.uk/averkiou/">Melinos Averkiou</a> &emsp;
    <a href="https://people.cs.umass.edu/~kalo/">Evangelos Kalogerakis</a>  &emsp;
</p>
<br>

<div align="center">
    <img src="images/teaser.png" width="100%" height ="50%" alt="teaser.png" />
</div>
<p align = 'center'>
    <small>The output probability per point can be used in pairwise terms to improve graph-based semantic segmentation methods 
    (left) by localizing boundaries between semantic parts. It can also be used in the geometric decomposition of point clouds
    into regions enclosed by sharp boundaries detected by our method (right).</small>
</p>


<h1 align = "center">Abstract</h1> 
We present a method that detects boundaries of parts in 3D shapes represented as point clouds. Our method is based on a 
graph convolutional network architecture that outputs a probability for a point to lie in an area that separates two or 
more parts in a 3D shape. Our boundary detector is quite generic: it can be trained to localize boundaries of semantic parts 
or geometric primitives commonly used in 3D modeling. Our experiments demonstrate that our method can extract more accurate
boundaries that are closer to ground-truth ones compared to alternatives. We also demonstrate an application of our network 
to fine-grained semantic shape segmentation, where we also show improvements in terms of part labeling performance.

<h1 align = "center">Boundary Datasets</h1>

<h3>Geometric segmentation dataset</h3>
<div align="center">
    <img src="images/abc_data.png" width="100%" height ="50%" alt="teaser.png" />
</div>
<p align = 'center'>
    <small>Marked (with red) boundaries on ABC point clouds for training.</small>
</p>


<h3>Semantic segmentation dataset</h3>
<div align="center">
    <img src="images/partnet_data.png" width="100%" height ="50%" alt="partnet_data.png" />
</div>
<p align = 'center'>
    <small>Marked boundaries on PartNet point clouds for training.</small>
</p>

__Datasets will be released soon__

<h1 align = "center">Results</h1>

<h3> Geometric part boundary detection </h3>
<div align="center">
    <img src="images/geometric_boundaries.png" width="100%" height ="40%" alt="geometric_boundaries.png" />
</div>
<p align = 'center'>
    <small>Boundaries detected by our method PB-DGCNN on some example ABC point clouds. The first column
    on the left shows the ground truth boundaries. The second column shows boundary probabilities produced 
    by PB-DGCNN, and the third column shows boundaries predicted by PB-DGCNN after thresholding.</small>
</p>

<h3>Semantic shape segmentation</h3>
<div align="center">
    <img src="images/semantic_segmentation.png" width="100%" height ="50%" alt="semantic_segmentation.png" />
</div>
<p align = 'center'>
    <small>Visual comparison of semantic segmentation for example PartNet point clouds, using DGCNN alone (unary), a graph-cut 
    formulation with normal angles in the pairwise term, and a graph-cut formulation with a combination of normal angles and
    boundary probabilities produced by PB-DGCNN in the pairwise term.</small>
</p>

 
