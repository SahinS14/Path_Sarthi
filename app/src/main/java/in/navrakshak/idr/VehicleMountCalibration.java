package in.navrakshak.idr;

import android.hardware.SensorManager;

/**
 * Learns a fixed vehicle-forward vector in device coordinates from trusted
 * GNSS course observations.  It therefore handles portrait/landscape holders
 * and pitch/roll, rather than assuming a particular phone axis is forward.
 */
final class VehicleMountCalibration {
    private final double[] forwardDevice = new double[3];
    private int samples;

    boolean ready() { return samples >= 12; }

    void observe(float[] rotationVector, double vehicleHeadingRad) {
        if (rotationVector == null || rotationVector.length < 3 || !Double.isFinite(vehicleHeadingRad)) return;
        float[] r = new float[9]; SensorManager.getRotationMatrixFromVector(r, rotationVector);
        // Android's rotation matrix transforms a device-frame vector into the
        // local East/North/Up world frame. Convert GNSS course back to device.
        double east = Math.sin(vehicleHeadingRad), north = Math.cos(vehicleHeadingRad);
        double x = r[0]*east + r[3]*north;
        double y = r[1]*east + r[4]*north;
        double z = r[2]*east + r[5]*north;
        double norm = Math.sqrt(x*x+y*y+z*z); if (norm < .8) return;
        x/=norm; y/=norm; z/=norm;
        if (samples == 0) { forwardDevice[0]=x; forwardDevice[1]=y; forwardDevice[2]=z; }
        else {
            double gain = .08; forwardDevice[0]=(1-gain)*forwardDevice[0]+gain*x; forwardDevice[1]=(1-gain)*forwardDevice[1]+gain*y; forwardDevice[2]=(1-gain)*forwardDevice[2]+gain*z;
            normalise(forwardDevice);
        }
        samples++;
    }

    double forwardAcceleration(float[] acceleration, float[] gravity) {
        if (!ready()) return acceleration[1]-gravity[1];
        return (acceleration[0]-gravity[0])*forwardDevice[0] + (acceleration[1]-gravity[1])*forwardDevice[1] + (acceleration[2]-gravity[2])*forwardDevice[2];
    }

    double vehicleHeading(float[] rotationVector) {
        if (!ready() || rotationVector == null || rotationVector.length < 3) return Double.NaN;
        float[] r = new float[9]; SensorManager.getRotationMatrixFromVector(r, rotationVector);
        double east=r[0]*forwardDevice[0]+r[1]*forwardDevice[1]+r[2]*forwardDevice[2];
        double north=r[3]*forwardDevice[0]+r[4]*forwardDevice[1]+r[5]*forwardDevice[2];
        return Math.atan2(east,north);
    }

    double verticalYawRate(float[] rotationVector, float[] gyro) {
        if (!ready() || rotationVector == null || rotationVector.length < 3) return gyro[2];
        float[] r = new float[9]; SensorManager.getRotationMatrixFromVector(r, rotationVector);
        // World vertical represented in device coordinates is R-transpose*[0,0,1].
        return gyro[0]*r[6]+gyro[1]*r[7]+gyro[2]*r[8];
    }

    private static void normalise(double[] v) { double n=Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); if(n>0){v[0]/=n;v[1]/=n;v[2]/=n;} }
}
