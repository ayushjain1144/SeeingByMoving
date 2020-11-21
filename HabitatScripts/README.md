# HabitatScripts
matterport_manual: Loads matterport3D meshes in habitat and launches the agent. Then you manually move around the scene and store sensor data.

<br>
replica_manual_aabb: Same as above but with replica data.

<br>
sim_automated_multiview.py:
Script which generates multiview data for objects in replica. 

<br>
sim_automated_cirle.py:
Loads replica in habitat and generated multiview data for objects in replica while keeping the camera fixated at the center of the object.

Command: python HabitatScripts/sim_automated_multiview.py

Place HabitatScripts in habitat-labs directory
