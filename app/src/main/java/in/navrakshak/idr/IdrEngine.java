package in.navrakshak.idr;

/** Framework-free 10 Hz GNSS+INS state engine, ported for on-device use. */
public final class IdrEngine {
    public static final class State {
        public double eastM, northM, speedMps, headingRad;
        public String mode = "WAITING_FOR_GNSS";
    }
    private final State state = new State();
    private long lastNs = 0;
    private double accBias, gyroBias, positionVariance = 100, speedVariance = 16;

    public State step(long timeNs, double forwardAcc, double yawRate, double vibration,
                      Double east, Double north, Double speed, Double heading, Double accuracy) {
        if (lastNs == 0) {
            lastNs = timeNs;
            if (east != null && north != null) initialise(east, north, speed, heading);
            return state;
        }
        double dt = Math.max(0.001, Math.min((timeNs - lastNs) / 1e9, 0.25));
        lastNs = timeNs;
        // IMU normally arrives before the first GNSS callback. Initialise as
        // soon as that first valid fix arrives instead of remaining stuck.
        if (state.mode.equals("WAITING_FOR_GNSS")) {
            if (east != null && north != null) initialise(east, north, speed, heading);
            return state;
        }
        double quality = Math.max(.05, 1.0 - Math.min(1.0, vibration));
        state.speedMps = Math.max(0, state.speedMps + (forwardAcc - accBias) * quality * dt);
        state.headingRad += (yawRate - gyroBias) * dt;
        state.eastM += state.speedMps * Math.sin(state.headingRad) * dt;
        state.northM += state.speedMps * Math.cos(state.headingRad) * dt;
        positionVariance += (.6 + 5 * vibration) * dt;
        speedVariance += (.15 + vibration) * dt;
        boolean gnssGood = east != null && north != null && accuracy != null && accuracy <= 30;
        if (!gnssGood) {
            // Rotation-vector heading is deliberately applied gently; it limits
            // yaw drift but does not pretend GNSS remains available.
            if (heading != null && state.speedMps > 0.5) state.headingRad += .08 * wrap(heading - state.headingRad);
            state.mode = "DEAD_RECKONING";
            return state;
        }
        double measurementVariance = Math.max(4, accuracy * accuracy);
        double gain = Math.max(.75, positionVariance / (positionVariance + measurementVariance));
        state.eastM += gain * (east - state.eastM); state.northM += gain * (north - state.northM);
        positionVariance *= 1 - gain;
        if (speed != null) {
            double speedGain = Math.max(.8, speedVariance / (speedVariance + 1));
            double residual = speed - state.speedMps;
            state.speedMps += speedGain * residual; accBias -= .01 * residual / Math.max(dt, .05); speedVariance *= 1 - speedGain;
        }
        if (heading != null && state.speedMps > 2) {
            double error = wrap(heading - state.headingRad); state.headingRad += .6 * error; gyroBias -= .002 * error / Math.max(dt, .05);
        }
        state.mode = "GNSS_AIDED"; return state;
    }
    private void initialise(double east, double north, Double speed, Double heading) {
        state.eastM = east; state.northM = north; state.speedMps = speed == null ? 0 : Math.max(0, speed);
        state.headingRad = heading == null ? 0 : heading; state.mode = "GNSS_AIDED";
    }
    private static double wrap(double radians) { return Math.atan2(Math.sin(radians), Math.cos(radians)); }
}
