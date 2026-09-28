package in.navrakshak.idr;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.speech.tts.TextToSpeech;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.drawable.GradientDrawable;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.os.Bundle;
import android.os.SystemClock;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageButton;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.core.view.ViewCompat;
import org.json.JSONArray;
import org.json.JSONObject;
import org.maplibre.android.MapLibre;
import org.maplibre.android.annotations.Marker;
import org.maplibre.android.annotations.MarkerOptions;
import org.maplibre.android.annotations.Icon;
import org.maplibre.android.annotations.IconFactory;
import org.maplibre.android.annotations.Polyline;
import org.maplibre.android.annotations.PolylineOptions;
import org.maplibre.android.camera.CameraUpdateFactory;
import org.maplibre.android.camera.CameraPosition;
import org.maplibre.android.geometry.LatLng;
import org.maplibre.android.maps.MapLibreMap;
import org.maplibre.android.maps.MapView;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.ArrayList;
import java.util.List;

public final class MainActivity extends androidx.activity.ComponentActivity implements SensorEventListener, LocationListener {
    private static final String TAG = "PathSarathiNav";
    private static final int LOCATION_REQUEST = 26168;
    private static final long GNSS_FRESHNESS_NS = 2_000_000_000L;
    // These colours describe the source of a displayed trace, not confidence or accuracy.
    private static final int PLANNED_ROUTE_COLOR = Color.rgb(94, 146, 232);
    private static final int GNSS_TRACK_COLOR = PLANNED_ROUTE_COLOR;
    private static final int ESTIMATED_TRACK_COLOR = Color.rgb(235, 195, 110);
    private static final float GNSS_RENDER_MIN_STEP_M = 1.0f;
    private static final float ESTIMATED_RENDER_MIN_STEP_M = 2.0f;
    private SensorManager sensors; private LocationManager locations; private final VehicleStateFilter engine = new VehicleStateFilter();
    private final Handler ui = new Handler(Looper.getMainLooper());
    private TextView mode, coordinates, metrics, logStatus, guidance, mapHint; private AppUi.StatusChip navigationChip; private Button startStop, outageButton, routeButton; private ImageButton locateButton; private EditText destinationInput; private MapView mapView; private MapLibreMap liveMap;
    private boolean recording, simulateOutage, hasRotation; private float[] accel = new float[3], gravity = new float[3], gyro = new float[3], rotation = new float[4];
    private Location latestLocation, latestGpsLocation, latestNetworkLocation; private long latestGpsReceiptNs, latestNetworkReceiptNs, locationSequence, consumedLocationSequence; private double originLat, originLon; private boolean hasOrigin; private FileWriter log;
    // Calibrated from GNSS course while moving: handles yaw, pitch and roll.
    private final VehicleMountCalibration mountCalibration = new VehicleMountCalibration();
    private OfflineRoadMap roadGraph; private Marker vehicleMarker, tripStartMarker, destinationMarker; private Polyline routeLine, activeGnssTrackLine, activeEstimatedTrackLine; private Icon vehicleIcon, startIcon, destinationIcon; private int vehicleIconHeadingBucket = Integer.MIN_VALUE; private boolean mapCentered, mapStyleReady, mapRenderFailureShown; private double displayEast, displayNorth; private long lastCameraUpdateNs; private java.util.List<LatLng> activeRoute = new java.util.ArrayList<>();
    // Keep observed GNSS and estimated paths as independent segments.  A segment is
    // never extended across a GNSS outage, so the map never invents a connection.
    private final List<List<LatLng>> gnssTrackSegments = new ArrayList<>();
    private final List<List<LatLng>> estimatedTrackSegments = new ArrayList<>();
    private final List<Polyline> gnssTrackLines = new ArrayList<>();
    private final List<Polyline> estimatedTrackLines = new ArrayList<>();
    private List<LatLng> activeGnssTrack, activeEstimatedTrack;
    private LatLng lastGnssTrackPoint, tripStartPoint;
    private long lastPlottedGnssMeasurementNs = Long.MIN_VALUE;
    private SpeedTcnInference speedModel; private double mlSpeedDelta = Double.NaN; private final double[] forwardHistory = new double[20]; private int forwardSamples, consecutiveTickFailures; private long outageStartedNs; private double tripDistanceM, lastTripEast, lastTripNorth; private boolean hasTripPoint; private long outageDurationNs, outageSegmentStartNs; private LatLng selectedDestination; private final List<GuideStep> guideSteps=new ArrayList<>(); private int guideIndex; private TextToSpeech speaker; private boolean speakerReady;

    @Override public void onCreate(Bundle saved) {
        super.onCreate(saved); MapLibre.getInstance(this); roadGraph = new OfflineRoadMap(this); speedModel = new SpeedTcnInference(this); speaker=new TextToSpeech(this,status->{speakerReady=status==TextToSpeech.SUCCESS; if(speakerReady)speaker.setLanguage(Locale.US);}); buildUi(); if (getIntent().getBooleanExtra("open_simulation", false)) logStatus.setText("Simulation ready: start a trip, acquire GPS, then choose Test outage."); sensors = (SensorManager) getSystemService(SENSOR_SERVICE); locations = (LocationManager) getSystemService(LOCATION_SERVICE);
        getOnBackPressedDispatcher().addCallback(this,new androidx.activity.OnBackPressedCallback(true){@Override public void handleOnBackPressed(){leaveNavigation();}});
    }
    private int dp(int value) { return (int)(value * getResources().getDisplayMetrics().density + .5f); }
    private TextView text(String value, int size) { TextView v = new TextView(this); v.setText(value); v.setTextSize(size); v.setTextColor(Color.WHITE); v.setPadding(dp(10), dp(4), dp(10), dp(4)); return v; }
    private GradientDrawable background(int color, float radius) { GradientDrawable shape = new GradientDrawable(); shape.setColor(color); shape.setCornerRadius(dp((int)radius)); return shape; }
    private void styleButton(Button button, int color) { button.setTextColor(Color.WHITE); button.setTextSize(13); button.setAllCaps(false); button.setBackground(background(color, 18)); button.setPadding(dp(12), 0, dp(12), 0); }
    private void buildUi() {
        LinearLayout root=AppUi.column(this);
        root.setBackgroundColor(AppUi.BG);
        LinearLayout top=AppUi.column(this);top.setPadding(dp(16),dp(10),dp(16),dp(12));
        LinearLayout bar=AppUi.row(this);
        ImageButton back=AppUi.iconButton(this,R.drawable.ic_arrow_back,"Back to dashboard",this::leaveNavigation);bar.addView(back,new LinearLayout.LayoutParams(dp(48),dp(48)));
        TextView title=AppUi.title(this,getIntent().getBooleanExtra("open_simulation",false)?"Outage test":"Live navigation",19);bar.addView(title,AppUi.weighted(this,12));
        navigationChip=AppUi.statusChip(this,"GPS SETUP",AppUi.AMBER);bar.addView(navigationChip);
        top.addView(bar);AppUi.gap(top,12);
        LinearLayout destinationRow=AppUi.row(this);
        destinationInput=new EditText(this);destinationInput.setHint("Search a destination");destinationInput.setContentDescription("Destination search");destinationInput.setTextColor(AppUi.TEXT);destinationInput.setHintTextColor(AppUi.MUTED);destinationInput.setTextSize(14);destinationInput.setSingleLine(true);destinationInput.setPadding(dp(14),dp(8),dp(14),dp(8));destinationInput.setBackground(AppUi.outlinedShape(this,AppUi.PANEL,14,AppUi.BORDER));destinationInput.setImeOptions(android.view.inputmethod.EditorInfo.IME_ACTION_SEARCH);
        destinationRow.addView(destinationInput,new LinearLayout.LayoutParams(0,dp(52),1));
        routeButton=AppUi.button(this,getString(R.string.ui_search),true,this::submitDestination);destinationRow.addView(routeButton,AppUi.weighted(this,8));((LinearLayout.LayoutParams)routeButton.getLayoutParams()).weight=.42f;
        destinationInput.setOnEditorActionListener((v,action,event)->{if(action==android.view.inputmethod.EditorInfo.IME_ACTION_SEARCH){submitDestination();return true;}return false;});
        top.addView(destinationRow);root.addView(top);

        FrameLayout mapFrame=new FrameLayout(this);mapFrame.setBackgroundColor(AppUi.SURFACE);mapView=new MapView(this);mapView.onCreate(null);mapFrame.addView(mapView,new FrameLayout.LayoutParams(-1,-1));
        root.addView(mapFrame,new LinearLayout.LayoutParams(-1,0,1));
        mapView.getMapAsync(map->{
            liveMap=map;
            mapStyleReady=false;
            liveMap.addOnMapClickListener(point->{selectDestination(point);return true;});
            liveMap.setStyle("https://tiles.openfreemap.org/styles/dark", style -> {
                if (isDestroyed()) return;
                mapStyleReady=true;
                mapRenderFailureShown=false;
                rebuildTrajectoryLayers();
            });
        });
        mapView.addOnDidFailLoadingMapListener(error->{mapStyleReady=false;if(!isDestroyed()){logStatus.setText("Map unavailable. Check internet; sensor recording can still run.");setNavigationStatus("MAP OFFLINE",AppUi.AMBER,"Map unavailable · sensor logging still works");}});

        LinearLayout mapCallout=AppUi.column(this);mapCallout.setPadding(dp(13),dp(11),dp(13),dp(11));mapCallout.setBackground(AppUi.outlinedShape(this,0xED132337,14,AppUi.BORDER));
        TextView mapLabel=AppUi.text(this,"PATH SĀRATHI",10,AppUi.GOLD);mapLabel.setLetterSpacing(.1f);mapCallout.addView(mapLabel,new LinearLayout.LayoutParams(-1,-2));AppUi.gap(mapCallout,5);
        mapHint=AppUi.text(this,"Start a trip to acquire your current position",12,AppUi.TEXT);mapCallout.addView(mapHint,new LinearLayout.LayoutParams(-1,-2));AppUi.gap(mapCallout,9);
        LinearLayout trajectoryLegend=AppUi.row(this);trajectoryLegend.setContentDescription("Map trace legend: blue is observed GNSS and route guidance; gold is Path Sārathi estimated position");
        // Keep the on-map key intentionally short; its content description carries the
        // full source semantics without forcing a multi-line chip on narrow phones.
        AppUi.StatusChip gnssLegend=AppUi.statusChip(this,"GNSS",GNSS_TRACK_COLOR);trajectoryLegend.addView(gnssLegend);
        AppUi.StatusChip estimateLegend=AppUi.statusChip(this,"IDR",ESTIMATED_TRACK_COLOR);LinearLayout.LayoutParams estimateParams=new LinearLayout.LayoutParams(-2,-2);estimateParams.setMarginStart(dp(6));trajectoryLegend.addView(estimateLegend,estimateParams);
        mapCallout.addView(trajectoryLegend,new LinearLayout.LayoutParams(-1,-2));
        FrameLayout.LayoutParams calloutParams=new FrameLayout.LayoutParams(dp(250),-2,Gravity.START|Gravity.TOP);calloutParams.setMargins(dp(12),dp(12),0,dp(12));mapFrame.addView(mapCallout,calloutParams);
        locateButton=AppUi.iconButton(this,R.drawable.ic_my_location,"Center map on vehicle",this::centerOnVehicle);
        FrameLayout.LayoutParams locate=new FrameLayout.LayoutParams(dp(50),dp(50),Gravity.END|Gravity.TOP);locate.setMargins(dp(12),dp(12),dp(12),dp(12));mapFrame.addView(locateButton,locate);

        android.widget.ScrollView statusScroll=new AppUi.StatusPanel(this);statusScroll.setBackground(AppUi.topSheet(this));statusScroll.setFillViewport(false);ViewCompat.setElevation(statusScroll,dp(8));
        LinearLayout bottom=AppUi.column(this);bottom.setPadding(dp(18),dp(17),dp(18),dp(16));statusScroll.addView(bottom);
        mode=AppUi.title(this,getString(R.string.ui_ready_to_start),18);bottom.addView(mode);
        AppUi.gap(bottom,7);
        coordinates=AppUi.text(this,getString(R.string.ui_distance_outage),13,AppUi.MUTED);bottom.addView(coordinates);AppUi.gap(bottom,6);
        metrics=AppUi.text(this,getString(R.string.ui_speed),14,AppUi.TEXT);bottom.addView(metrics);AppUi.gap(bottom,6);
        guidance=AppUi.text(this,getString(R.string.ui_search_or_tap_the_map_to_choose_a),13,AppUi.GOLD);bottom.addView(guidance);AppUi.gap(bottom,8);
        logStatus=AppUi.text(this,getString(R.string.ui_start_outdoors_to_acquire_a_gps_fix),12,AppUi.MUTED);logStatus.setAccessibilityLiveRegion(View.ACCESSIBILITY_LIVE_REGION_POLITE);bottom.addView(logStatus);AppUi.gap(bottom,14);
        LinearLayout controls=AppUi.row(this);
        startStop=AppUi.button(this,getString(R.string.ui_start_trip),true,this::toggleRecording);controls.addView(startStop,AppUi.weighted(this,0));
        outageButton=AppUi.button(this,getString(R.string.ui_test_outage),false,this::toggleOutage);controls.addView(outageButton,AppUi.weighted(this,9));bottom.addView(controls);
        root.addView(statusScroll,new LinearLayout.LayoutParams(-1,-2));
        // Short windows use a side panel so controls never consume the map.
        if(getResources().getConfiguration().screenHeightDp<500){root.removeView(mapFrame);root.removeView(statusScroll);LinearLayout landscape=AppUi.row(this);landscape.addView(mapFrame,new LinearLayout.LayoutParams(0,-1,1));landscape.addView(statusScroll,new LinearLayout.LayoutParams(dp(Math.min(340,getResources().getConfiguration().screenWidthDp/2)),-1));root.addView(landscape,new LinearLayout.LayoutParams(-1,0,1));}
        AppUi.install(this,root);
    }
    private void setNavigationStatus(String label,int color,String detail){
        if(navigationChip!=null)navigationChip.setState(label,color);
        if(mapHint!=null)mapHint.setText(detail);
    }
    private void submitDestination(){((android.view.inputmethod.InputMethodManager)getSystemService(INPUT_METHOD_SERVICE)).hideSoftInputFromWindow(destinationInput.getWindowToken(),0);destinationInput.clearFocus();if(destinationInput.getText().toString().trim().isEmpty()&&selectedDestination!=null)requestRoute(selectedDestination);else searchDestination();}
    private void toggleOutage(){
        if(simulateOutage){simulateOutage=false;outageStartedNs=0;outageButton.setText("Test outage");logStatus.setText("Simulation ended. Waiting for a fresh GPS fix.");setNavigationStatus("ACQUIRING GPS",AppUi.AMBER,"Simulation ended · waiting for a fresh GNSS fix");return;}
        if(!recording||!hasOrigin){logStatus.setText("Start a trip and acquire GPS before simulating an outage.");return;}
        new android.app.AlertDialog.Builder(this).setTitle("Simulate a GNSS outage?").setMessage("GPS fixes will be withheld from the navigation filter. The displayed position will be an experimental inertial estimate. Use this test while stationary or as a passenger.").setNegativeButton("Cancel",null).setPositiveButton("Start test",(d,w)->{simulateOutage=true;outageStartedNs=SystemClock.elapsedRealtimeNanos();outageButton.setText("Restore GPS");logStatus.setText("Simulation active · GPS is intentionally withheld.");setNavigationStatus("SIMULATION",AppUi.AMBER,"GNSS is withheld · showing an experimental estimate");}).show();
    }
    private void leaveNavigation(){if(recording)new android.app.AlertDialog.Builder(this).setTitle("Save and end this trip?").setMessage("Your sensor log will be saved to the trip library.").setNegativeButton("Keep recording",null).setPositiveButton("Save & leave",(d,w)->{stopRecording();finish();}).show();else finish();}
    private void toggleRecording() { if (recording) stopRecording(); else startRecording(); }
    private void startRecording() {
        if (recording) return;
        if (checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) { requestPermissions(new String[]{Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION}, LOCATION_REQUEST); return; }
        if(sensors.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)==null || sensors.getDefaultSensor(Sensor.TYPE_GYROSCOPE)==null || sensors.getDefaultSensor(Sensor.TYPE_GRAVITY)==null || sensors.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)==null){logStatus.setText("Required motion sensors are unavailable. Check Settings & sensors.");return;}
        updateScreenAwake(true);setNavigationStatus("ACQUIRING GPS",AppUi.AMBER,"Looking for a fresh outdoor GNSS fix");
        // A new trip must never inherit a timestamp, velocity, or neural-history
        // window from the previous one. Carrying that state over can make a new
        // outage appear frozen before the first trusted GNSS measurement arrives.
        engine.reset();
        speedModel.reset();
        java.util.Arrays.fill(forwardHistory, 0.0);
        forwardSamples=0;
        mlSpeedDelta=Double.NaN;
        consecutiveTickFailures=0;
        mapRenderFailureShown=false;
        recording = true; tripDistanceM=0;hasTripPoint=false;outageDurationNs=0;outageSegmentStartNs=0; clearTrajectoryVisuals(); mapCentered=false; lastCameraUpdateNs=0; startStop.setText("Save & stop"); register(Sensor.TYPE_ACCELEROMETER); register(Sensor.TYPE_GRAVITY); register(Sensor.TYPE_GYROSCOPE); register(Sensor.TYPE_ROTATION_VECTOR);
        boolean gpsEnabled = locations.isProviderEnabled(LocationManager.GPS_PROVIDER);
        boolean networkEnabled = locations.isProviderEnabled(LocationManager.NETWORK_PROVIDER);
        if (!gpsEnabled && !networkEnabled) { logStatus.setText("Location disabled: enable Location/GPS in phone settings"); }
        try {
            if (gpsEnabled) { locations.requestLocationUpdates(LocationManager.GPS_PROVIDER, 500, 0, this); seedLastKnown(LocationManager.GPS_PROVIDER); }
            if (networkEnabled) { locations.requestLocationUpdates(LocationManager.NETWORK_PROVIDER, 500, 0, this); seedLastKnown(LocationManager.NETWORK_PROVIDER); }
        } catch (SecurityException error) { logStatus.setText("Location permission denied. Open App info → Permissions → Location."); }
        openLog(); ui.post(tick);
    }
    private void seedLastKnown(String provider) { try { Location candidate = locations.getLastKnownLocation(provider); if (candidate != null) onLocationChanged(candidate); } catch (SecurityException ignored) { } }
    private void stopRecording() { if(!recording)return;if(outageSegmentStartNs>0){outageDurationNs+=SystemClock.elapsedRealtimeNanos()-outageSegmentStartNs;outageSegmentStartNs=0;} recording=false;ui.removeCallbacks(tick);simulateOutage=false;outageStartedNs=0;outageButton.setText("Test outage");startStop.setText("Start trip");sensors.unregisterListener(this);locations.removeUpdates(this);closeLog();updateScreenAwake(false);mode.setText("Trip ended");mode.setTextColor(AppUi.TEXT);logStatus.setText(String.format(Locale.getDefault(),"Trip saved • %.2f km • GNSS outage %.1f s",tripDistanceM/1000,outageDurationNs/1e9));setNavigationStatus("TRIP SAVED",AppUi.GOLD,"Trip saved locally · review it in the trip library"); }
    private void register(int type) { Sensor sensor = sensors.getDefaultSensor(type); if (sensor != null) sensors.registerListener(this, sensor, SensorManager.SENSOR_DELAY_GAME); }
    private final Runnable tick = new Runnable() {
        @Override public void run() {
            if (!recording) return;
            try {
                advance(SystemClock.elapsedRealtimeNanos());
                consecutiveTickFailures=0;
            } catch (RuntimeException error) {
                consecutiveTickFailures++;
                // A MapLibre annotation failure must not silently kill the 10 Hz
                // navigation loop. The next tick is always scheduled below.
                if (consecutiveTickFailures == 1) {
                    Log.e(TAG, "Navigation update failed; keeping sensor loop alive", error);
                    logStatus.setText("Navigation display recovered; sensor tracking continues.");
                }
            } finally {
                if (recording) ui.postDelayed(this, 100);
            }
        }
    };
    private void advance(long ns) {
        Double east = null, north = null, speed = null, heading = null, accuracy = null;
        Location freshLocation = !simulateOutage ? freshLocation(ns) : null;
        boolean gnssAvailable = freshLocation != null && hasOrigin && LocationManager.GPS_PROVIDER.equals(freshLocation.getProvider());
        boolean newGnssMeasurement = gnssAvailable && locationSequence != consumedLocationSequence;
        if (gnssAvailable) {
            if (newGnssMeasurement) consumedLocationSequence = locationSequence;
            double[] xy = toLocal(freshLocation.getLatitude(), freshLocation.getLongitude()); east = xy[0]; north = xy[1]; speed = (double) freshLocation.getSpeed(); heading = Math.toRadians(freshLocation.getBearing()); accuracy = (double) freshLocation.getAccuracy();
            if (speed > 2.0f) mountCalibration.observe(rotation, heading);
        }
        // During an outage this is a low-confidence calibrated rotation-vector
        // correction, never a substitute for a GNSS position measurement.
        double calibratedHeading = mountCalibration.vehicleHeading(rotation);
        if ((simulateOutage || !gnssAvailable) && Double.isFinite(calibratedHeading)) heading = calibratedHeading;
        double forwardAcc=mountCalibration.forwardAcceleration(accel, gravity);
        double yawRate=mountCalibration.verticalYawRate(rotation, gyro);
        double vibration = Math.min(1, Math.abs(forwardAcc) / 8.0);
        // Contract navrakshak-speed-tcn-v1: acceleration XYZ, gravity XYZ,
        // then the legacy IO-VNBD yaw/pitch/roll ordering. This raw-axis model
        // is not mount-invariant; its dataset-frame equivalence remains
        // provisional and its delta must not be treated as absolute speed.
        float[] modelFeatures=SpeedTcnContract.fromAndroid(accel,gravity,gyro);
        mlSpeedDelta = speedModel.update(modelFeatures);
        for (int i=0;i<19;i++) forwardHistory[i]=forwardHistory[i+1]; forwardHistory[19]=forwardAcc; if(forwardSamples<20)forwardSamples++;
        double baseConfidence=1.0-vibration, integratedDelta=0; for(double sample:forwardHistory)integratedDelta+=sample*.1;
        boolean shortOutage=simulateOutage&&outageStartedNs>0&&(ns-outageStartedNs)<5_000_000_000L;
        boolean modelConsistent=!Double.isNaN(mlSpeedDelta)&&Math.abs(mlSpeedDelta-integratedDelta)<1.5;
        double imuConfidence=baseConfidence;
        if(shortOutage&&modelConsistent) imuConfidence*=Math.exp(-Math.abs(mlSpeedDelta-integratedDelta)/1.5);
        // A vehicle travelling straight at constant speed has almost zero
        // acceleration and yaw-rate.  That is not evidence that it is parked.
        // Only apply zero-velocity/bias learning while GNSS independently
        // confirms near-zero speed; never zero the state during an outage.
        double stationary = (gnssAvailable && freshLocation.getSpeed() < .35f
                && Math.abs(forwardAcc) < .12 && Math.abs(yawRate) < .04) ? 1.0 : 0.0;
        // The bundled graph is a demo extract, not a verified offline pack for
        // the user's current region. Never use it as a position pseudo-measurement:
        // it can pull a live trajectory onto an unrelated road and make it look stuck.
        VehicleStateFilter.State state = engine.step(ns, forwardAcc, yawRate, imuConfidence, stationary, gnssAvailable, newGnssMeasurement, east, north, speed, heading, accuracy, null, null, null, 0);
        if(hasTripPoint)tripDistanceM+=Math.hypot(state.eastM-lastTripEast,state.northM-lastTripNorth);lastTripEast=state.eastM;lastTripNorth=state.northM;hasTripPoint=true;
        if(state.mode.equals("DEAD_RECKONING")){if(outageSegmentStartNs==0)outageSegmentStartNs=ns;}else if(outageSegmentStartNs>0){outageDurationNs+=ns-outageSegmentStartNs;outageSegmentStartNs=0;}
        String label=state.mode.equals("GNSS_AIDED")?"GPS + inertial tracking":state.mode.equals("DEAD_RECKONING")?"Estimated position · GPS unavailable":"Acquiring GPS…";mode.setText(simulateOutage?"SIMULATION · Estimated position":label);mode.setTextColor(state.mode.equals("GNSS_AIDED")?AppUi.GREEN:AppUi.AMBER);
        if(simulateOutage)setNavigationStatus("SIMULATION",AppUi.AMBER,"GNSS withheld · displaying an experimental estimate");
        else if(state.mode.equals("GNSS_AIDED"))setNavigationStatus("GNSS LIVE",AppUi.GREEN,"GNSS and inertial measurements are updating the route");
        else if(state.mode.equals("DEAD_RECKONING"))setNavigationStatus("ESTIMATING",AppUi.AMBER,"GNSS unavailable · position is being estimated");
        else setNavigationStatus("ACQUIRING GPS",AppUi.AMBER,"Looking for a fresh outdoor GNSS fix");
        coordinates.setText(String.format(Locale.US, "Distance %.2f km  •  Outage %.1f s", tripDistanceM/1000, (outageDurationNs+(outageSegmentStartNs>0?ns-outageSegmentStartNs:0))/1e9)); metrics.setText(String.format(Locale.getDefault(),"%.1f km/h  ·  %s",state.speedMps*3.6,mountCalibration.ready()?"Mount aligned":"Learning mount alignment"));
        displayEast = state.eastM; displayNorth = state.northM;
        try {
            updateTrajectoryLayers(state, gnssAvailable ? freshLocation : null, newGnssMeasurement);
            updateMap(state);
        } catch (RuntimeException error) {
            reportMapRenderFailure(error);
        }
        writeLog(ns, state, east, north, accuracy, gnssAvailable ? freshLocation : null, modelFeatures);
    }
    @Override public void onSensorChanged(SensorEvent event) { if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) accel = event.values.clone(); else if (event.sensor.getType() == Sensor.TYPE_GRAVITY) gravity = event.values.clone(); else if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE) gyro = event.values.clone(); else if (event.sensor.getType() == Sensor.TYPE_ROTATION_VECTOR) { rotation = event.values.clone(); hasRotation = true; } }
    @Override public void onAccuracyChanged(Sensor sensor, int accuracy) { }
    @Override public void onLocationChanged(Location location) {
        // Network fixes can jump substantially on moving bikes. Prefer hardware
        // GPS as long as it is available; network remains only a startup fallback.
        long receiptNs = SystemClock.elapsedRealtimeNanos();
        long measurementNs = location.getElapsedRealtimeNanos();
        if (measurementNs <= 0 || receiptNs-measurementNs > GNSS_FRESHNESS_NS || receiptNs < measurementNs) {
            logStatus.setText("Ignoring stale location fix — acquiring current GNSS");
            return;
        }
        Location copy = new Location(location);
        if (LocationManager.GPS_PROVIDER.equals(location.getProvider())) { latestGpsLocation = copy; latestGpsReceiptNs = receiptNs; }
        else { latestNetworkLocation = copy; latestNetworkReceiptNs = receiptNs; }
        Location candidate = freshLocation(receiptNs);
        if (candidate != null) { latestLocation = candidate; locationSequence++; }
        if (!hasOrigin && latestLocation != null) { originLat = latestLocation.getLatitude(); originLon = latestLocation.getLongitude(); hasOrigin = true; roadGraph.setOrigin(originLat, originLon); }
        if (latestLocation != null && !simulateOutage) {logStatus.setText(String.format(Locale.US, "GNSS fix: ±%.1f m (%s)", latestLocation.getAccuracy(), latestLocation.getProvider()));setNavigationStatus("GNSS LIVE",AppUi.GREEN,"Fresh GNSS position acquired · start moving when ready");}
    }
    /** Prefer hardware GPS while it is recent; use network location only as an acquisition fallback. */
    private Location freshLocation(long nowNs) {
        if (latestGpsLocation != null && nowNs-latestGpsReceiptNs >= 0 && nowNs-latestGpsReceiptNs <= GNSS_FRESHNESS_NS) return latestGpsLocation;
        if (latestNetworkLocation != null && nowNs-latestNetworkReceiptNs >= 0 && nowNs-latestNetworkReceiptNs <= GNSS_FRESHNESS_NS) return latestNetworkLocation;
        return null;
    }
    @Override public void onProviderDisabled(String provider) { logStatus.setText(provider + " disabled — enable phone Location");setNavigationStatus("GPS SETUP",AppUi.AMBER,"Enable phone location to establish your starting position"); }
    @Override public void onProviderEnabled(String provider) { logStatus.setText(provider + " enabled — acquiring GNSS fix");setNavigationStatus("ACQUIRING GPS",AppUi.AMBER,"Location enabled · waiting for a fresh GNSS fix"); }
    private double[] toLocal(double lat, double lon) { double r = 6371000; double north = r * Math.toRadians(lat - originLat); double east = r * Math.toRadians(lon - originLon) * Math.cos(Math.toRadians(originLat)); return new double[]{east, north}; }
    private static boolean isValidCoordinate(double latitude, double longitude) {
        return Double.isFinite(latitude) && Double.isFinite(longitude)
                && Math.abs(latitude) <= 90 && Math.abs(longitude) <= 180;
    }

    private boolean isMapReady() {
        return liveMap != null && mapStyleReady && !isDestroyed();
    }

    private void reportMapRenderFailure(RuntimeException error) {
        if (!mapRenderFailureShown) {
            mapRenderFailureShown=true;
            Log.w(TAG, "Map rendering paused; navigation filter remains active", error);
            if (recording) logStatus.setText("Map redraw paused; sensor navigation is still running.");
        }
    }

    private void updateTrajectoryLayers(VehicleStateFilter.State state, Location gnssMeasurement, boolean newGnssMeasurement) {
        if (newGnssMeasurement && gnssMeasurement != null) {
            long measurementNs = gnssMeasurement.getElapsedRealtimeNanos();
            // The filter can be rendered many times per fix. Only a new hardware
            // GPS observation may extend the blue observed-GNSS trace.
            if ((measurementNs <= 0 || measurementNs > lastPlottedGnssMeasurementNs)
                    && isValidCoordinate(gnssMeasurement.getLatitude(), gnssMeasurement.getLongitude())) {
                LatLng observed = new LatLng(gnssMeasurement.getLatitude(), gnssMeasurement.getLongitude());
                appendGnssObservation(observed);
                if (measurementNs > 0) lastPlottedGnssMeasurementNs = measurementNs;
            }
        }
        if (gnssMeasurement == null) {
            // Close this observed segment. A later reacquisition starts another,
            // rather than drawing a blue GNSS line through an unavailable interval.
            activeGnssTrack = null;
            activeGnssTrackLine = null;
        }

        if ("DEAD_RECKONING".equals(state.mode)) {
            double[] geo = roadGraph.geo(state.eastM, state.northM);
            if (!isValidCoordinate(geo[0], geo[1])) return;
            LatLng estimated = new LatLng(geo[0], geo[1]);
            if (activeEstimatedTrack == null) {
                activeEstimatedTrack = new ArrayList<>();
                estimatedTrackSegments.add(activeEstimatedTrack);
                // Colour switches at the last actual GNSS point, but a future
                // outage creates a new amber segment instead of bridging history.
                if (lastGnssTrackPoint != null) activeEstimatedTrack.add(lastGnssTrackPoint);
            }
            // A 2 m visual spacing avoids overwhelming MapLibre during a long
            // blackout while the filter and CSV logger still run at 10 Hz.
            if (appendDistinctPoint(activeEstimatedTrack, estimated, ESTIMATED_RENDER_MIN_STEP_M)) {
                activeEstimatedTrackLine = upsertTrackLine(activeEstimatedTrackLine, activeEstimatedTrack, estimatedTrackLines, ESTIMATED_TRACK_COLOR, 8f);
            }
        } else {
            activeEstimatedTrack = null;
            activeEstimatedTrackLine = null;
        }
    }

    private void appendGnssObservation(LatLng observed) {
        lastGnssTrackPoint = observed;
        if (tripStartPoint == null) {
            tripStartPoint = observed;
            ensureTripStartMarker();
        }
        if (activeGnssTrack == null) {
            activeGnssTrack = new ArrayList<>();
            gnssTrackSegments.add(activeGnssTrack);
        }
        if (appendDistinctPoint(activeGnssTrack, observed, GNSS_RENDER_MIN_STEP_M)) {
            activeGnssTrackLine = upsertTrackLine(activeGnssTrackLine, activeGnssTrack, gnssTrackLines, GNSS_TRACK_COLOR, 7f);
        }
    }

    /** Reduces stationary duplicate points without adding interpolated observations. */
    private boolean appendDistinctPoint(List<LatLng> points, LatLng next, float minimumDistanceM) {
        if (points.isEmpty()) {
            points.add(next);
            return true;
        }
        LatLng previous = points.get(points.size() - 1);
        float[] distance = new float[1];
        Location.distanceBetween(previous.getLatitude(), previous.getLongitude(), next.getLatitude(), next.getLongitude(), distance);
        if (distance[0] < minimumDistanceM) return false;
        points.add(next);
        return true;
    }

    private Polyline upsertTrackLine(Polyline line, List<LatLng> points, List<Polyline> allLines, int color, float width) {
        if (!isMapReady() || points.size() < 2) return line;
        try {
            if (line == null) {
                PolylineOptions options = new PolylineOptions().width(width).color(color).alpha(.94f);
                options.addAll(points);
                line = liveMap.addPolyline(options);
                allLines.add(line);
            } else {
                // MapLibre's addPoint performs its own single native update. The
                // old full-list setPoints + updatePolyline pair duplicated work
                // and grew quadratically on a long dead-reckoning segment.
                line.addPoint(points.get(points.size() - 1));
            }
        } catch (RuntimeException error) {
            reportMapRenderFailure(error);
        }
        return line;
    }

    private void clearTrajectoryVisuals() {
        removeTrackLines(gnssTrackLines);
        removeTrackLines(estimatedTrackLines);
        gnssTrackSegments.clear();
        estimatedTrackSegments.clear();
        activeGnssTrack = null;
        activeEstimatedTrack = null;
        activeGnssTrackLine = null;
        activeEstimatedTrackLine = null;
        lastGnssTrackPoint = null;
        tripStartPoint = null;
        lastPlottedGnssMeasurementNs = Long.MIN_VALUE;
        try {
            if (isMapReady() && tripStartMarker != null) liveMap.removeMarker(tripStartMarker);
            if (isMapReady() && vehicleMarker != null) liveMap.removeMarker(vehicleMarker);
        } catch (RuntimeException error) {
            reportMapRenderFailure(error);
        }
        tripStartMarker = null;
        vehicleMarker = null;
        vehicleIconHeadingBucket = Integer.MIN_VALUE;
    }

    private void removeTrackLines(List<Polyline> lines) {
        try {
            if (isMapReady()) {
                for (Polyline line : lines) liveMap.removePolyline(line);
            }
        } catch (RuntimeException error) {
            reportMapRenderFailure(error);
        } finally {
            lines.clear();
        }
    }

    /** Rebuilds annotations if MapLibre recreates its style after data already exists. */
    private void rebuildTrajectoryLayers() {
        if (!isMapReady()) return;
        try {
            removeTrackLines(gnssTrackLines);
            removeTrackLines(estimatedTrackLines);
            activeGnssTrackLine = null;
            activeEstimatedTrackLine = null;
            for (List<LatLng> segment : gnssTrackSegments) {
                Polyline line = upsertTrackLine(null, segment, gnssTrackLines, GNSS_TRACK_COLOR, 7f);
                if (segment == activeGnssTrack) activeGnssTrackLine = line;
            }
            for (List<LatLng> segment : estimatedTrackSegments) {
                Polyline line = upsertTrackLine(null, segment, estimatedTrackLines, ESTIMATED_TRACK_COLOR, 8f);
                if (segment == activeEstimatedTrack) activeEstimatedTrackLine = line;
            }
            routeLine=null;
            destinationMarker=null;
            vehicleMarker=null;
            if (activeRoute.size() > 1) {
                PolylineOptions options = new PolylineOptions().width(5).color(PLANNED_ROUTE_COLOR).alpha(.55f);
                options.addAll(activeRoute);
                routeLine=liveMap.addPolyline(options);
            }
            if (selectedDestination != null) {
                destinationMarker=liveMap.addMarker(new MarkerOptions().position(selectedDestination).icon(destinationIcon()).title("Destination"));
            }
            tripStartMarker=null;
            ensureTripStartMarker();
        } catch (RuntimeException error) {
            reportMapRenderFailure(error);
        }
    }

    private void ensureTripStartMarker() {
        if (!isMapReady() || tripStartPoint == null || tripStartMarker != null) return;
        try {
            tripStartMarker = liveMap.addMarker(new MarkerOptions().position(tripStartPoint).icon(startIcon()).title("Trip start · GNSS observed"));
        } catch (RuntimeException error) {
            reportMapRenderFailure(error);
        }
    }

    private void updateMap(VehicleStateFilter.State state) {
        if (!isMapReady() || !hasOrigin) return;
        double[] geo = roadGraph.geo(state.eastM, state.northM);
        if (!isValidCoordinate(geo[0], geo[1])) return;
        LatLng point = new LatLng(geo[0], geo[1]);
        try {
            Icon icon = vehicleIcon(state.headingRad);
            String markerTitle = "GNSS_AIDED".equals(state.mode) ? Brand.NAME + " · GNSS observed" : Brand.NAME + " · estimated position";
            if (vehicleMarker == null) {
                vehicleMarker = liveMap.addMarker(new MarkerOptions().position(point).icon(icon).title(markerTitle));
            } else {
                // Marker mutators already update MapLibre. Do not send a third
                // explicit update on every sensor tick.
                vehicleMarker.setPosition(point);
                if (vehicleMarker.getIcon() != icon) vehicleMarker.setIcon(icon);
                if (!markerTitle.equals(vehicleMarker.getTitle())) vehicleMarker.setTitle(markerTitle);
            }
            ensureTripStartMarker();
            updateGuidance(point);
            if (!mapCentered) { liveMap.moveCamera(CameraUpdateFactory.newLatLngZoom(point, 17)); mapCentered = true; }
            long now=SystemClock.elapsedRealtimeNanos();
            if (recording && now-lastCameraUpdateNs>1_250_000_000L) {
                // Keep the map north-up: the arrow bitmap is the explicit and stable
                // vehicle-heading indicator, even when a user manually recentres.
                CameraPosition camera = new CameraPosition.Builder().target(point).zoom(17).bearing(0).tilt(35).build();
                liveMap.animateCamera(CameraUpdateFactory.newCameraPosition(camera), 600);
                lastCameraUpdateNs=now;
            }
        } catch (RuntimeException error) {
            reportMapRenderFailure(error);
        }
    }

    /** Annotation markers do not expose a heading property, so rotate a compact bitmap safely. */
    private Icon vehicleIcon(double headingRad) {
        int heading = Double.isFinite(headingRad) ? (int) Math.round(Math.toDegrees(headingRad)) : 0;
        heading = ((heading % 360) + 360) % 360;
        int bucket = ((heading + 5) / 10) * 10;
        if (vehicleIcon != null && bucket == vehicleIconHeadingBucket) return vehicleIcon;
        Bitmap bitmap=Bitmap.createBitmap(84,84,Bitmap.Config.ARGB_8888);
        Canvas canvas=new Canvas(bitmap);
        Paint paint=new Paint(Paint.ANTI_ALIAS_FLAG);
        paint.setColor(Color.WHITE); canvas.drawCircle(42,42,32,paint);
        paint.setColor(Color.rgb(37,99,235)); canvas.drawCircle(42,42,28,paint);
        paint.setColor(Color.WHITE);
        Path arrow=new Path(); arrow.moveTo(42,12); arrow.lineTo(21,60); arrow.lineTo(42,51); arrow.lineTo(63,60); arrow.close();
        canvas.save(); canvas.rotate(bucket,42,42); canvas.drawPath(arrow,paint); canvas.restore();
        vehicleIcon=IconFactory.getInstance(this).fromBitmap(bitmap);
        vehicleIconHeadingBucket=bucket;
        return vehicleIcon;
    }

    private Icon startIcon() {
        if (startIcon != null) return startIcon;
        Bitmap bitmap=Bitmap.createBitmap(64,64,Bitmap.Config.ARGB_8888);
        Canvas canvas=new Canvas(bitmap); Paint paint=new Paint(Paint.ANTI_ALIAS_FLAG);
        paint.setColor(Color.WHITE); canvas.drawCircle(32,32,24,paint);
        paint.setColor(GNSS_TRACK_COLOR); canvas.drawCircle(32,32,20,paint);
        paint.setColor(Color.WHITE); Path play=new Path(); play.moveTo(27,21); play.lineTo(45,32); play.lineTo(27,43); play.close(); canvas.drawPath(play,paint);
        startIcon=IconFactory.getInstance(this).fromBitmap(bitmap);
        return startIcon;
    }

    private Icon destinationIcon() {
        if (destinationIcon != null) return destinationIcon;
        Bitmap bitmap=Bitmap.createBitmap(64,64,Bitmap.Config.ARGB_8888);
        Canvas canvas=new Canvas(bitmap); Paint paint=new Paint(Paint.ANTI_ALIAS_FLAG);
        paint.setColor(Color.WHITE); canvas.drawCircle(32,32,25,paint);
        paint.setColor(AppUi.BG); canvas.drawCircle(32,32,21,paint);
        paint.setStrokeWidth(4); paint.setColor(Color.WHITE); canvas.drawLine(23,16,23,48,paint);
        int square=8;
        for(int row=0;row<2;row++) for(int column=0;column<3;column++) {
            paint.setColor(((row+column)&1)==0 ? Color.WHITE : ESTIMATED_TRACK_COLOR);
            canvas.drawRect(23+column*square,16+row*square,23+(column+1)*square,16+(row+1)*square,paint);
        }
        destinationIcon=IconFactory.getInstance(this).fromBitmap(bitmap);
        return destinationIcon;
    }
    private void centerOnVehicle() {
        if (!isMapReady() || !hasOrigin) { logStatus.setText("Waiting for map and GNSS position"); return; }
        double[] geo = roadGraph.geo(displayEast, displayNorth);
        if (!isValidCoordinate(geo[0], geo[1])) return;
        try {
            liveMap.animateCamera(CameraUpdateFactory.newLatLngZoom(new LatLng(geo[0], geo[1]), 17));
        } catch (RuntimeException error) {
            reportMapRenderFailure(error);
        }
    }
    private void searchDestination() { String query = destinationInput.getText().toString().trim(); if (query.isEmpty()) { logStatus.setText("Enter a destination, or tap a point on the map"); return; } logStatus.setText("Searching destination…"); new Thread(() -> { try { String url = "https://nominatim.openstreetmap.org/search?format=jsonv2&limit=1&q=" + URLEncoder.encode(query, "UTF-8"); JSONArray matches = getJson(url); if (matches.length() == 0) throw new IOException("No matching place found"); JSONObject match = matches.getJSONObject(0); LatLng point = new LatLng(Double.parseDouble(match.getString("lat")), Double.parseDouble(match.getString("lon"))); ui.post(() -> requestRoute(point)); } catch (Exception error) { ui.post(() -> logStatus.setText("Destination search failed — check Internet or tap map")); } }).start(); }
    /** First map tap only pins the target. This avoids accidental route calls while panning. */
    private void selectDestination(LatLng destination) {
        if (!isMapReady()) { logStatus.setText("Map is still loading. Try again in a moment."); return; }
        selectedDestination = destination;
        placeDestinationMarker(destination, "Pinned destination");
        destinationInput.setText("");
        // The compact action slot intentionally uses one word so it stays
        // readable on a narrow Android screen.
        routeButton.setText("Route");
        logStatus.setText("Destination pinned — tap Route to confirm");
        if(mapHint!=null)mapHint.setText("Destination pinned · build a route to begin guidance");
    }

    private void placeDestinationMarker(LatLng destination, String title) {
        if (!isMapReady()) return;
        try {
            if (destinationMarker != null) liveMap.removeMarker(destinationMarker);
            destinationMarker = liveMap.addMarker(new MarkerOptions().position(destination).icon(destinationIcon()).title(title));
        } catch (RuntimeException error) {
            reportMapRenderFailure(error);
        }
    }

    private void requestRoute(LatLng destination) { if (!isMapReady() || !hasOrigin) { logStatus.setText("Start a trip and wait for map and GPS before building a route."); return; } selectedDestination = destination; placeDestinationMarker(destination, "Destination"); double[] source = roadGraph.geo(displayEast, displayNorth); logStatus.setText("Calculating driving route…"); new Thread(() -> { try { String url = String.format(Locale.US, "https://router.project-osrm.org/route/v1/driving/%.7f,%.7f;%.7f,%.7f?overview=full&geometries=geojson&steps=true", source[1], source[0], destination.getLongitude(), destination.getLatitude()); JSONObject route = getJsonObject(url).getJSONArray("routes").getJSONObject(0); JSONArray coordinates = route.getJSONObject("geometry").getJSONArray("coordinates"); ArrayList<LatLng> path = new ArrayList<>(); for (int i=0;i<coordinates.length();i++) { JSONArray point=coordinates.getJSONArray(i); path.add(new LatLng(point.getDouble(1), point.getDouble(0))); } ArrayList<GuideStep> steps=new ArrayList<>(); JSONArray rawSteps=route.getJSONArray("legs").getJSONObject(0).getJSONArray("steps"); for(int i=0;i<rawSteps.length();i++){JSONObject s=rawSteps.getJSONObject(i),m=s.getJSONObject("maneuver");JSONArray loc=m.getJSONArray("location");String type=m.optString("type","continue"), modifier=m.optString("modifier","");String road=s.optString("name","");String instruction=(type.equals("arrive")?"Arrive at destination":(modifier.isEmpty()?"Continue":"Turn "+modifier))+(road.isEmpty()?"":" onto "+road);steps.add(new GuideStep(new LatLng(loc.getDouble(1),loc.getDouble(0)),instruction,s.optDouble("distance",0)));} ui.post(() -> drawRoute(path,steps)); } catch (Exception error) { ui.post(() -> logStatus.setText("Route failed — check Internet or choose another destination")); } }).start(); }
    private void drawRoute(List<LatLng> path,List<GuideStep> steps) { if (!isMapReady() || path.size() < 2) { logStatus.setText("No drivable route found"); return; } activeRoute = new ArrayList<>(path);guideSteps.clear();guideSteps.addAll(steps);guideIndex=0; try { if (routeLine != null) liveMap.removePolyline(routeLine); PolylineOptions options = new PolylineOptions().width(5).color(PLANNED_ROUTE_COLOR).alpha(.55f); options.addAll(path); routeLine = liveMap.addPolyline(options); guidance.setText("Guidance: route ready — follow the blue route"); say("Route ready. Follow the highlighted route."); logStatus.setText("Route ready — blue route remains visible during GNSS outage");if(mapHint!=null)mapHint.setText("Route ready · observed and estimated traces remain distinct"); } catch (RuntimeException error) { reportMapRenderFailure(error); } }
    private static final class GuideStep { final LatLng location;final String instruction;final double afterMeters;GuideStep(LatLng l,String i,double d){location=l;instruction=i;afterMeters=d;} }
    private void updateGuidance(LatLng vehicle){if(guideSteps.isEmpty()||guideIndex>=guideSteps.size())return;GuideStep step=guideSteps.get(guideIndex);float[] result=new float[1];Location.distanceBetween(vehicle.getLatitude(),vehicle.getLongitude(),step.location.getLatitude(),step.location.getLongitude(),result);if(result[0]<24&&guideIndex<guideSteps.size()-1){guideIndex++;step=guideSteps.get(guideIndex);say(step.instruction);}guidance.setText(String.format(Locale.US,"Next: %s • %.0f m",step.instruction,result[0]));}
    private void say(String message){if(speakerReady&&AppUi.prefs(this).getBoolean("voice",true))speaker.speak(message,TextToSpeech.QUEUE_FLUSH,null,"path-sarathi-guide");}
    private JSONArray getJson(String address) throws Exception { HttpURLConnection connection=(HttpURLConnection)new URL(address).openConnection(); connection.setRequestProperty("User-Agent","PathSarathi-IDR/1.0"); connection.setConnectTimeout(12000); connection.setReadTimeout(12000); try(BufferedReader reader=new BufferedReader(new InputStreamReader(connection.getInputStream()))){StringBuilder body=new StringBuilder();String line;while((line=reader.readLine())!=null)body.append(line);return new JSONArray(body.toString());} finally {connection.disconnect();} }
    private JSONObject getJsonObject(String address) throws Exception { HttpURLConnection connection=(HttpURLConnection)new URL(address).openConnection(); connection.setRequestProperty("User-Agent","PathSarathi-IDR/1.0"); connection.setConnectTimeout(12000); connection.setReadTimeout(12000); try(BufferedReader reader=new BufferedReader(new InputStreamReader(connection.getInputStream()))){StringBuilder body=new StringBuilder();String line;while((line=reader.readLine())!=null)body.append(line);return new JSONObject(body.toString());} finally {connection.disconnect();} }
    private Double deviceHeading() { if (!hasRotation || rotation.length < 3) return null; float[] matrix = new float[9]; float[] angles = new float[3]; SensorManager.getRotationMatrixFromVector(matrix, rotation); SensorManager.getOrientation(matrix, angles); return wrap(angles[0]); }
    private static double wrap(double radians) { return Math.atan2(Math.sin(radians), Math.cos(radians)); }
    private void openLog() { try { File dir = getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS); if (dir != null) { File f = new File(dir, Brand.LOG_PREFIX + new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date()) + ".csv"); log = new FileWriter(f); log.write("time_ns,mode,east_m,north_m,speed_mps,gnss_east_m,gnss_north_m,gnss_accuracy_m,gnss_lat_deg,gnss_lon_deg,gnss_speed_mps,gnss_bearing_deg,raw_gnss_accuracy_m,raw_provider,acc_x,acc_y,acc_z,gravity_x,gravity_y,gravity_z,gyro_x,gyro_y,gyro_z,phone_heading_rad,mount_calibrated,model_f0,model_f1,model_f2,model_f3,model_f4,model_f5,model_f6,model_f7,model_f8,model_speed_delta_mps\n"); logStatus.setText("Logger: " + f.getName()); } } catch (IOException e) { logStatus.setText("Logger error: " + e.getMessage()); } }
    private void writeLog(long ns, VehicleStateFilter.State s, Double east, Double north, Double accuracy, Location freshGnss, float[] modelFeatures) { if (log == null) return; try { Double lat = freshGnss == null ? null : freshGnss.getLatitude(); Double lon = freshGnss == null ? null : freshGnss.getLongitude(); Double rawSpeed = freshGnss == null ? null : (double) freshGnss.getSpeed(); Double bearing = freshGnss == null ? null : (double) freshGnss.getBearing(); Double rawAccuracy = freshGnss == null ? null : (double) freshGnss.getAccuracy(); String provider = freshGnss == null ? "null" : freshGnss.getProvider(); Double phoneHeading = deviceHeading(); log.write(String.format(Locale.US, "%d,%s,%.5f,%.5f,%.5f,%s,%s,%s,%s,%s,%s,%s,%s,%s,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%s,%s,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%s\n", ns,s.mode,s.eastM,s.northM,s.speedMps,east,north,accuracy,lat,lon,rawSpeed,bearing,rawAccuracy,provider,accel[0],accel[1],accel[2],gravity[0],gravity[1],gravity[2],gyro[0],gyro[1],gyro[2],phoneHeading,mountCalibration.ready() ? 1.0 : null,modelFeatures[0],modelFeatures[1],modelFeatures[2],modelFeatures[3],modelFeatures[4],modelFeatures[5],modelFeatures[6],modelFeatures[7],modelFeatures[8],Double.isNaN(mlSpeedDelta) ? null : mlSpeedDelta)); } catch (IOException ignored) {} }
    private void closeLog() { try { if (log != null) { log.flush(); log.close(); } } catch (IOException ignored) {} log = null; }
    @Override public void onRequestPermissionsResult(int request, String[] permissions, int[] results) { super.onRequestPermissionsResult(request, permissions, results); if (request == LOCATION_REQUEST && results.length > 0 && results[0] == PackageManager.PERMISSION_GRANTED) startRecording(); else if (request == LOCATION_REQUEST) {logStatus.setText("Precise location is needed. Allow it in Settings to start a trip.");new android.app.AlertDialog.Builder(this).setTitle("Precise location required").setMessage("Enable precise location to establish the starting position and record your trip.").setNegativeButton("Not now",null).setPositiveButton("Open settings",(d,w)->startActivity(new android.content.Intent(android.provider.Settings.ACTION_APPLICATION_DETAILS_SETTINGS,android.net.Uri.parse("package:"+getPackageName())))).show();} }
    @Override protected void onStart() { super.onStart(); if (mapView != null) mapView.onStart(); }
    private void updateScreenAwake(boolean active){if(active&&AppUi.prefs(this).getBoolean("keep_awake",true))getWindow().addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);else getWindow().clearFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);}
    @Override protected void onResume() { super.onResume(); if (mapView != null) mapView.onResume();updateScreenAwake(recording); }
    @Override protected void onPause() { if (mapView != null) mapView.onPause(); super.onPause(); }
    @Override protected void onStop() { if(recording)stopRecording();if (mapView != null) mapView.onStop(); super.onStop(); }
    @Override public void onLowMemory() { super.onLowMemory(); if (mapView != null) mapView.onLowMemory(); }
    @Override protected void onDestroy() { stopRecording();ui.removeCallbacksAndMessages(null); if(speaker!=null){speaker.stop();speaker.shutdown();} if (mapView != null) mapView.onDestroy(); super.onDestroy(); }
}
