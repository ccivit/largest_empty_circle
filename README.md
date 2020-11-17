# Largest Empty Circle (or inscribed circle)

This script demonstrates one solution to the problem of finding the largest empty circle in the plane whose interior does not overlap with any given point. This is done through the use of a Voronoi diagram, with a faster convergence than trivial computation and guaranteed to converge accurately.

This script was used to demonstrate that the methods used by a major industrial computer vision manufacturer were incorrect due to implemented heuristics. The example provided was found in the wild and made their algorithm provide innacurate results. I informed their application engineers and new versions of their $100k microscope now gives accurate results for this set (I do not have confirmation if they implemented this method).

Voronoi diagrams, when used for this problem, provide a candidate set of centers, which is guaranteed to contain the center (defined as the position of maximum distance to all points).

![](example.png?raw=true)
