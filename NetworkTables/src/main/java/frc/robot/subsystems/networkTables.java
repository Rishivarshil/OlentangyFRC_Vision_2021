// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.


/*
*************************** 
Things need to Updated:
- Gyro Angle
- Location of Camera on Bot
- Send Position
***************************
*/

package frc.robot.subsystems;

import edu.wpi.first.wpilibj2.command.SubsystemBase;
import edu.wpi.first.networktables.NetworkTable;
import edu.wpi.first.networktables.NetworkTableEntry;
import edu.wpi.first.math.Matrix;
import edu.wpi.first.math.Nat;
import edu.wpi.first.math.VecBuilder;
import edu.wpi.first.math.numbers.N1;
import edu.wpi.first.math.numbers.N3;
import edu.wpi.first.wpilibj.smartdashboard.SmartDashboard;
import edu.wpi.first.networktables.EntryListenerFlags;
import edu.wpi.first.networktables.NetworkTableInstance;


public class networkTables extends SubsystemBase {
  /** Listens for changes in the NetworkTables. */
  public networkTables() {}

  @Override
  public void periodic() {
    NetworkTableInstance inst = NetworkTableInstance.getDefault();
    listenTvecs(inst);
   
  }

  public void listenTvecs(NetworkTableInstance inst){
    NetworkTable Tvecs = inst.getTable("SmartDashboard"); //delcares the networktables to the already intizialized instance

    inst.startClientTeam(190);
    listener(Tvecs);    
    
    try {
       Thread.sleep(10000);
    } catch (InterruptedException ex) {
       System.out.println("Interrupted");
       Thread.currentThread().interrupt();
       return;
    }
  }

  public void listener(NetworkTable Tvecs){
    NetworkTableEntry Time = Tvecs.getEntry("Time");
    NetworkTableEntry XEntry = Tvecs.getEntry("X");
    NetworkTableEntry YEntry = Tvecs.getEntry("Y");
    NetworkTableEntry ZEntry = Tvecs.getEntry("Z");
    //adds an entry listener for changed values of "Time", the lambda ("->" operator)
    Time.addListener(event -> {
      System.out.println("Time changed value: " + event.getEntry().getValue()); //If time is changed, then it records x,y,z
      System.out.println("X changed value: " + XEntry.getValue());
      double x = XEntry.getDouble(0);
      System.out.println("Y changed value: " + YEntry.getValue());
      double y = YEntry.getDouble(0);
      System.out.println("Z changed value: " + ZEntry.getValue());
      double z = ZEntry.getDouble(0);

      var position = convertTvec(x, y, z); //converts the x,y,z into final position vector
      // Do work here like updates odometry...
      System.out.print(position);

    }, EntryListenerFlags.kNew | EntryListenerFlags.kUpdate);
  }

  public Matrix<N3, N1> convertTvec(double x, double y, double z) {

    // Build out input vector
    var in_vec = VecBuilder.fill(x, y, z);

    var cv2_correction_mat = Matrix.mat(Nat.N3(), Nat.N3()).fill(
      0, 0, 1,
      -1, 0, 0,
      0, -1, 0);

    var corrected_vec = cv2_correction_mat.times(in_vec);

    // Build our rotation matrix
    var pitch = -10 * Math.PI / 180;
    var c = Math.cos(pitch);
    var s = Math.sin(pitch);
    var camera_to_bot = Matrix.mat(Nat.N3(), Nat.N3()).fill(
       c, 0, s,
       0, 1, 0,
      -s, 0, c);

    // Rotate
    var corrected_bot_oriented = camera_to_bot.times(corrected_vec);


    var trans2_vec = VecBuilder.fill(0, 0, 1); //Change this where we know the displacement of the camera to the center of the robot

    var out_vec = trans2_vec.plus(corrected_bot_oriented);
    double gyro_angle = 1.0; //add the heading by multipling by gyro angle
    var final_vec = out_vec.times(gyro_angle);
    
    var hub_coordinates = VecBuilder.fill(457, 134.5, 0);

    var position_vec =  hub_coordinates.minus(final_vec);
    // Print output
    SmartDashboard.putNumber("out_x", final_vec.get(0, 0));
    SmartDashboard.putNumber("out_y", final_vec.get(1, 0));
    SmartDashboard.putNumber("out_z", final_vec.get(2, 0)); 

    return position_vec; 
  }

  @Override
  public void simulationPeriodic() {
    periodic();
  }
}