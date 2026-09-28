package in.navrakshak.idr;

/** Exact input contract for navrakshak-speed-tcn-v1. Units are SI. */
final class SpeedTcnContract {
    static final int SAMPLE_HZ=10, WINDOW_SAMPLES=20;
    static final String OUTPUT_SEMANTICS="speed_delta_mps_over_2s";
    static final String[] FEATURES={"acc_x_mps2","acc_y_mps2","acc_z_mps2","gravity_x_mps2","gravity_y_mps2","gravity_z_mps2","gyro_yaw_rps","gyro_pitch_rps","gyro_roll_rps"};
    private SpeedTcnContract() { }
    /** Provisional IO-VNBD yaw/pitch/roll compatibility mapping: Android z,y,x. */
    static float[] fromAndroid(float[] acceleration,float[] gravity,float[] gyro) {
        if(acceleration.length<3||gravity.length<3||gyro.length<3) throw new IllegalArgumentException("Expected Android XYZ sensors");
        return new float[]{acceleration[0],acceleration[1],acceleration[2],gravity[0],gravity[1],gravity[2],gyro[2],gyro[1],gyro[0]};
    }
}
