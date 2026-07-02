#!/bin/bash
source /opt/ros/foxy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=1
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/root/f1tenth_ws/src:/root/f1tenth_ws/src/racecar_description/models
export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:/root/f1tenth_ws/install/gazebo_ros2_2dmap_plugin/lib

for track in monza_track interlagos_track silverstone_track; do
    echo "=========================================="
    echo "Processing $track..."
    mkdir -p src/f1tenth_lattice_ros2/maps/$track
    
    if [ "$track" == "monza_track" ]; then
        INIT_X="-10.0"
        INIT_Y="5.0"
        MAP_H="0.2"
    elif [ "$track" == "silverstone_track" ]; then
        INIT_X="-10.0"
        INIT_Y="2.0"
        MAP_H="0.2"
    elif [ "$track" == "interlagos_track" ]; then
        INIT_X="2.0"
        INIT_Y="-10.0"
        MAP_H="0.2"
    fi
    
    # Inject plugin with correct starting point dynamically
    sed '/<\/world>/i \ \ <plugin name="gazebo_occupancy_map" filename="libgazebo_2Dmap_plugin.so"><map_resolution>0.05</map_resolution><map_height>'$MAP_H'</map_height><init_robot_x>'$INIT_X'</init_robot_x><init_robot_y>'$INIT_Y'</init_robot_y></plugin>' src/racecar_description/worlds/${track}.world > /tmp/temp_${track}.world
    
    gzserver /tmp/temp_${track}.world &
    GZ_PID=$!
    
    echo "Waiting 10 seconds for Gazebo and models to load..."
    sleep 10
    
    echo "Starting map saver in background..."
    python3 src/f1tenth_lattice_ros2/tools/save_map.py $track &
    SAVER_PID=$!
    
    echo "Calling generate_map service..."
    ros2 service call /gazebo_2Dmap_plugin/generate_map std_srvs/srv/Empty
    
    echo "Waiting for map saver to receive and save the map..."
    wait $SAVER_PID
    
    echo "Killing Gazebo..."
    kill $GZ_PID
    wait $GZ_PID 2>/dev/null
    echo "$track map generated successfully."
done
echo "All maps generated!"
