package in.navrakshak.idr;

/**
 * Portable 2D vehicle-state GNSS/INS filter. It keeps only forward vehicle
 * motion, enforcing the non-holonomic constraint that a road vehicle does not
 * independently move sideways.
 */
public final class VehicleStateFilter {
    // Selected on the provisional IO-VNBD session split. It prevents raw phone
    // acceleration bias from overwhelming the last trusted speed in a blackout.
    private static final double OUTAGE_ACCELERATION_GAIN = 0.10;

    public static final class State {
        public double eastM, northM, speedMps, headingRad, accBiasMps2, gyroBiasRps;
        public double positionSigmaM = 25, speedSigmaMps = 4;
        public String mode = "WAITING_FOR_GNSS";
    }

    private final State state = new State();
    private long lastNs;
    private double stationarySeconds;

    /** Clears all per-trip state while retaining the filter object itself. */
    public void reset() {
        state.eastM=0;
        state.northM=0;
        state.speedMps=0;
        state.headingRad=0;
        state.accBiasMps2=0;
        state.gyroBiasRps=0;
        state.positionSigmaM=25;
        state.speedSigmaMps=4;
        state.mode="WAITING_FOR_GNSS";
        lastNs=0;
        stationarySeconds=0;
    }

    public State step(long timeNs, double forwardAcc, double yawRate, double imuConfidence,
                      double stationaryProbability, boolean gnssAvailable,
                      boolean newGnssMeasurement, Double gnssEast, Double gnssNorth,
                      Double gnssSpeed, Double gnssHeading, Double gnssAccuracy,
                      Double routeEast, Double routeNorth, Double routeHeading,
                      double routeConfidence) {
        forwardAcc=finiteOr(forwardAcc, 0);
        yawRate=finiteOr(yawRate, 0);
        imuConfidence=finiteOr(imuConfidence, .02);
        stationaryProbability=finiteOr(stationaryProbability, 0);

        if (lastNs == 0) {
            lastNs=timeNs;
            if (finite(gnssEast) && finite(gnssNorth)) {
                initialise(gnssEast, gnssNorth, gnssSpeed, gnssHeading);
            }
            return state;
        }

        double dt=Math.max(.001, Math.min((timeNs-lastNs)/1e9, .25));
        lastNs=timeNs;
        if (state.mode.equals("WAITING_FOR_GNSS")) {
            if (finite(gnssEast) && finite(gnssNorth)) {
                initialise(gnssEast, gnssNorth, gnssSpeed, gnssHeading);
            }
            return state;
        }

        boolean hasGnss=gnssAvailable && finite(gnssEast) && finite(gnssNorth)
                && finite(gnssAccuracy) && gnssAccuracy <= 30;
        double confidence=clamp(imuConfidence, .02, 1);
        stationarySeconds=stationaryProbability > .8 ? stationarySeconds+dt : 0;

        // Propagation in the vehicle-forward direction is the 2D NHC: there is
        // no free lateral velocity state to accumulate sideways drift. During a
        // blackout, suppress most raw MEMS acceleration integration. The app
        // retains the last trusted GNSS speed rather than dropping to zero from
        // a small negative mount/accelerometer bias.
        double outageGain=hasGnss ? 1.0 : OUTAGE_ACCELERATION_GAIN;
        double acceleration=(forwardAcc-state.accBiasMps2)*confidence*outageGain;
        state.speedMps=Math.max(0, state.speedMps+acceleration*dt);
        state.headingRad=wrap(state.headingRad+(yawRate-state.gyroBiasRps)*dt);
        state.eastM+=state.speedMps*Math.sin(state.headingRad)*dt;
        state.northM+=state.speedMps*Math.cos(state.headingRad)*dt;
        state.positionSigmaM=Math.min(500, state.positionSigmaM+(.25+3*(1-confidence))*dt);
        state.speedSigmaMps=Math.min(30, state.speedSigmaMps+(.04+.8*(1-confidence))*dt);

        // A single trusted GNSS observation may trigger a zero-velocity update.
        // Repeating speed*=.25 for every 10 Hz render tick was able to collapse
        // speed to zero just before an outage began.
        if (hasGnss && newGnssMeasurement && stationarySeconds > .5) {
            state.accBiasMps2=clamp(state.accBiasMps2+.08*forwardAcc, -3, 3);
            state.speedMps*=.25;
            state.speedSigmaMps=Math.max(.05, state.speedSigmaMps*.6);
        }

        // A receiver sample is fused at most once; intermediate IMU steps still
        // propagate the state, which keeps GNSS-aided motion continuous.
        if (hasGnss && newGnssMeasurement) {
            gnssUpdate(gnssEast, gnssNorth, gnssSpeed, gnssHeading, gnssAccuracy, dt);
        }
        state.mode=hasGnss ? "GNSS_AIDED" : "DEAD_RECKONING";

        // Road constraints can gently align heading only. Position projection is
        // deliberately excluded here: an unverified or region-mismatched map
        // pack must never teleport or pin a real vehicle's state.
        if (finite(routeHeading) && routeConfidence >= .85 && state.speedMps > 1.0) {
            double gain=.10*Math.min(1, routeConfidence);
            state.headingRad=wrap(state.headingRad+gain*wrap(routeHeading-state.headingRad));
        }

        recoverFiniteState();
        return state;
    }

    private void initialise(double east, double north, Double speed, Double heading) {
        state.eastM=east;
        state.northM=north;
        state.speedMps=Math.max(0, finiteOr(speed, 0));
        state.headingRad=finiteOr(heading, 0);
        state.mode="GNSS_AIDED";
    }

    private void gnssUpdate(double east, double north, Double speed, Double heading,
                            double accuracy, double dt) {
        double sigma=Math.max(2, accuracy);
        double gain=state.positionSigmaM*state.positionSigmaM
                /(state.positionSigmaM*state.positionSigmaM+sigma*sigma);
        state.eastM+=gain*(east-state.eastM);
        state.northM+=gain*(north-state.northM);
        state.positionSigmaM*=Math.sqrt(1-gain);
        if (finite(speed)) {
            double speedGain=state.speedSigmaMps*state.speedSigmaMps
                    /(state.speedSigmaMps*state.speedSigmaMps+.49);
            double residual=speed-state.speedMps;
            state.speedMps+=speedGain*residual;
            state.accBiasMps2=clamp(state.accBiasMps2-.02*residual/Math.max(.05, dt), -3, 3);
            state.speedSigmaMps=Math.max(.05, state.speedSigmaMps*Math.sqrt(1-speedGain));
        }
        if (finite(heading) && state.speedMps > 2) {
            double error=wrap(heading-state.headingRad);
            state.headingRad=wrap(state.headingRad+.55*error);
            state.gyroBiasRps=clamp(state.gyroBiasRps-.001*error/Math.max(.05, dt), -2, 2);
        }
    }

    private void recoverFiniteState() {
        if (!Double.isFinite(state.eastM)) state.eastM=0;
        if (!Double.isFinite(state.northM)) state.northM=0;
        if (!Double.isFinite(state.speedMps) || state.speedMps < 0) state.speedMps=0;
        if (!Double.isFinite(state.headingRad)) state.headingRad=0;
        if (!Double.isFinite(state.accBiasMps2)) state.accBiasMps2=0;
        if (!Double.isFinite(state.gyroBiasRps)) state.gyroBiasRps=0;
        if (!Double.isFinite(state.positionSigmaM) || state.positionSigmaM < 0) state.positionSigmaM=25;
        if (!Double.isFinite(state.speedSigmaMps) || state.speedSigmaMps < 0) state.speedSigmaMps=4;
    }

    private static boolean finite(Double value) {
        return value != null && Double.isFinite(value);
    }

    private static double finiteOr(double value, double fallback) {
        return Double.isFinite(value) ? value : fallback;
    }

    private static double finiteOr(Double value, double fallback) {
        return finite(value) ? value : fallback;
    }

    private static double clamp(double value, double lower, double upper) {
        return Math.max(lower, Math.min(upper, value));
    }

    private static double wrap(double radians) {
        return Math.atan2(Math.sin(radians), Math.cos(radians));
    }
}
