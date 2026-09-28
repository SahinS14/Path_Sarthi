package in.navrakshak.idr;

import android.app.Activity;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.view.Gravity;
import android.widget.*;

import org.maplibre.android.MapLibre;
import org.maplibre.android.annotations.Marker;
import org.maplibre.android.annotations.MarkerOptions;
import org.maplibre.android.annotations.PolylineOptions;
import org.maplibre.android.camera.CameraUpdateFactory;
import org.maplibre.android.geometry.LatLng;
import org.maplibre.android.geometry.LatLngBounds;
import org.maplibre.android.maps.MapLibreMap;
import org.maplibre.android.maps.MapView;

import java.io.*;
import java.util.*;

/**
 * Replays the sources that were actually written to a trip CSV. Recorded GNSS
 * samples are never presented as ground truth, and a gold trace is drawn only
 * for rows the engine logged in DEAD_RECKONING mode.
 */
public final class ReplayActivity extends Activity {
    private static final double EARTH_RADIUS_M = 6_371_000.0;
    private static final int GPS_ROUTE_COLOR = android.graphics.Color.rgb(94, 146, 232);
    private static final int IDR_TRACK_COLOR = android.graphics.Color.rgb(235, 195, 110);

    private enum Source { RAW_GNSS, DEAD_RECKONING, FILTERED_STATE }

    private static final class ReplayPoint {
        final long timeNs;
        final long order;
        final LatLng position;
        final Source source;

        ReplayPoint(long timeNs, long order, LatLng position, Source source) {
            this.timeNs = timeNs;
            this.order = order;
            this.position = position;
            this.source = source;
        }
    }

    private static final class Sample {
        final long timeNs;
        final long order;
        final String mode;
        final Double eastM;
        final Double northM;
        final LatLng rawGnss;

        Sample(long timeNs, long order, String mode, Double eastM, Double northM, LatLng rawGnss) {
            this.timeNs = timeNs;
            this.order = order;
            this.mode = mode;
            this.eastM = eastM;
            this.northM = northM;
            this.rawGnss = rawGnss;
        }
    }

    private static final class LoadedTrip {
        final List<ReplayPoint> timeline = new ArrayList<>();
        final List<List<LatLng>> rawGnssSegments = new ArrayList<>();
        final List<List<LatLng>> deadReckoningSegments = new ArrayList<>();
        int rawGnssCount;
        int deadReckoningCount;
        boolean legacyRawGnssOnly;
        LatLng start;
        LatLng end;
    }

    private final List<ReplayPoint> timeline = new ArrayList<>();
    private final List<List<LatLng>> rawGnssSegments = new ArrayList<>();
    private final List<List<LatLng>> deadReckoningSegments = new ArrayList<>();
    private final Handler handler = new Handler(Looper.getMainLooper());
    private MapView mapView;
    private MapLibreMap map;
    private Marker vehicleMarker, startMarker, endMarker;
    private TextView status, pointDetail;
    private Button play;
    private SeekBar seek;
    private LatLng tripStart, tripEnd;
    private int index;
    private boolean playing, mapReady, loaded, drawn;

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        MapLibre.getInstance(this);
        build(state);
        String path = getIntent().getStringExtra("csv_path");
        new Thread(() -> load(path), "trip-reader").start();
    }

    private void load(String path) {
        LoadedTrip trip = new LoadedTrip();
        String error = null;
        try {
            File documents = getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS);
            if (path == null || documents == null) throw new IOException();
            File file = new File(path).getCanonicalFile();
            // A replay can only read an application-owned log immediately inside Documents.
            if (!documents.getCanonicalFile().equals(file.getParentFile())) throw new IOException();
            parseTrip(file, trip);
        } catch (Exception ignored) {
            error = "This file cannot be replayed. It may be missing or use an unsupported CSV format.";
        }

        final LoadedTrip loadedTrip = trip;
        final String message = error;
        handler.post(() -> {
            if (isFinishing() || isDestroyed()) return;
            timeline.addAll(loadedTrip.timeline);
            rawGnssSegments.addAll(loadedTrip.rawGnssSegments);
            deadReckoningSegments.addAll(loadedTrip.deadReckoningSegments);
            tripStart = loadedTrip.start;
            tripEnd = loadedTrip.end;
            loaded = true;
            status.setText(message != null ? message : summary(loadedTrip));
            pointDetail.setText(timeline.isEmpty() ? "No replay points available" : "0 / " + timeline.size() + " points");
            seek.setMax(Math.max(0, timeline.size() - 1));
            render();
        });
    }

    private static void parseTrip(File file, LoadedTrip trip) throws IOException {
        List<Sample> samples = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String header = reader.readLine();
            if (header == null) throw new IOException();
            Map<String, Integer> columns = columns(header);
            int latitudeColumn = column(columns, "gnss_lat_deg", "lat_deg", "latitude", "lat");
            int longitudeColumn = column(columns, "gnss_lon_deg", "lon_deg", "longitude", "lon");
            if (latitudeColumn < 0 || longitudeColumn < 0) throw new IOException();

            int timeColumn = column(columns, "time_ns", "timestamp_ns", "time");
            int modeColumn = column(columns, "mode");
            int eastColumn = column(columns, "east_m");
            int northColumn = column(columns, "north_m");
            trip.legacyRawGnssOnly = modeColumn < 0 || eastColumn < 0 || northColumn < 0;

            String line;
            long order = 0;
            while ((line = reader.readLine()) != null) {
                if (Thread.currentThread().isInterrupted()) return;
                String[] fields = line.split(",", -1);
                LatLng rawGnss = coordinateAt(fields, latitudeColumn, longitudeColumn);
                Double east = numberAt(fields, eastColumn);
                Double north = numberAt(fields, northColumn);
                if (rawGnss == null && (east == null || north == null)) {
                    order++;
                    continue;
                }
                String mode = textAt(fields, modeColumn);
                samples.add(new Sample(timeAt(fields, timeColumn, order), order, mode, east, north, rawGnss));
                order++;
            }
        }

        Collections.sort(samples, (left, right) -> {
            int timeOrder = Long.compare(left.timeNs, right.timeNs);
            return timeOrder != 0 ? timeOrder : Long.compare(left.order, right.order);
        });
        if (samples.isEmpty()) return;

        LatLng localOrigin = localOrigin(samples);
        List<LatLng> currentRawGnss = null;
        List<LatLng> currentDeadReckoning = null;
        for (Sample sample : samples) {
            LatLng estimated = stateCoordinate(localOrigin, sample.eastM, sample.northM);
            boolean deadReckoning = "DEAD_RECKONING".equalsIgnoreCase(sample.mode);

            if (sample.rawGnss != null) {
                if (currentRawGnss == null) {
                    currentRawGnss = new ArrayList<>();
                    trip.rawGnssSegments.add(currentRawGnss);
                }
                currentRawGnss.add(sample.rawGnss);
                trip.rawGnssCount++;
            } else {
                currentRawGnss = null;
            }

            if (deadReckoning && estimated != null) {
                if (currentDeadReckoning == null) {
                    currentDeadReckoning = new ArrayList<>();
                    trip.deadReckoningSegments.add(currentDeadReckoning);
                }
                currentDeadReckoning.add(estimated);
                trip.deadReckoningCount++;
            } else {
                currentDeadReckoning = null;
            }

            ReplayPoint point = null;
            if (deadReckoning && estimated != null) {
                point = new ReplayPoint(sample.timeNs, sample.order, estimated, Source.DEAD_RECKONING);
            } else if (sample.rawGnss != null) {
                point = new ReplayPoint(sample.timeNs, sample.order, sample.rawGnss, Source.RAW_GNSS);
            } else if (estimated != null && !trip.legacyRawGnssOnly) {
                // Cursor fallback only: it is not added to either coloured trace.
                point = new ReplayPoint(sample.timeNs, sample.order, estimated, Source.FILTERED_STATE);
            }
            if (point != null) trip.timeline.add(point);
        }

        if (!trip.timeline.isEmpty()) {
            trip.start = trip.timeline.get(0).position;
            trip.end = trip.timeline.get(trip.timeline.size() - 1).position;
        }
    }

    private static Map<String, Integer> columns(String header) {
        Map<String, Integer> result = new HashMap<>();
        String[] values = header.split(",", -1);
        for (int i = 0; i < values.length; i++) result.put(normalizeHeader(values[i]), i);
        return result;
    }

    private static int column(Map<String, Integer> columns, String... names) {
        for (String name : names) {
            Integer index = columns.get(name);
            if (index != null) return index;
        }
        return -1;
    }

    private static String normalizeHeader(String value) {
        return value.replace("\uFEFF", "").trim().toLowerCase(Locale.US);
    }

    private static String textAt(String[] fields, int column) {
        return column < 0 || column >= fields.length ? "" : fields[column].trim();
    }

    private static Double numberAt(String[] fields, int column) {
        if (column < 0 || column >= fields.length) return null;
        String value = fields[column].trim();
        if (value.isEmpty() || "null".equalsIgnoreCase(value)) return null;
        try {
            double parsed = Double.parseDouble(value);
            return Double.isFinite(parsed) ? parsed : null;
        } catch (NumberFormatException ignored) {
            return null;
        }
    }

    private static long timeAt(String[] fields, int column, long fallback) {
        if (column < 0 || column >= fields.length) return fallback;
        try {
            return Long.parseLong(fields[column].trim());
        } catch (NumberFormatException ignored) {
            return fallback;
        }
    }

    private static LatLng coordinateAt(String[] fields, int latitudeColumn, int longitudeColumn) {
        Double latitude = numberAt(fields, latitudeColumn);
        Double longitude = numberAt(fields, longitudeColumn);
        if (latitude == null || longitude == null || Math.abs(latitude) > 90 || Math.abs(longitude) > 180) return null;
        return new LatLng(latitude, longitude);
    }

    /** Derive the app's local EN origin from a row containing both state and raw GNSS. */
    private static LatLng localOrigin(List<Sample> samples) {
        for (Sample sample : samples) {
            if (sample.rawGnss == null || sample.eastM == null || sample.northM == null) continue;
            double latitude = sample.rawGnss.getLatitude() - Math.toDegrees(sample.northM / EARTH_RADIUS_M);
            double longitude = sample.rawGnss.getLongitude() - Math.toDegrees(
                    sample.eastM / (EARTH_RADIUS_M * Math.cos(Math.toRadians(latitude))));
            if (Double.isFinite(latitude) && Double.isFinite(longitude)
                    && Math.abs(latitude) <= 90 && Math.abs(longitude) <= 180) {
                return new LatLng(latitude, longitude);
            }
        }
        for (Sample sample : samples) if (sample.rawGnss != null) return sample.rawGnss;
        return null;
    }

    private static LatLng stateCoordinate(LatLng origin, Double eastM, Double northM) {
        if (origin == null || eastM == null || northM == null) return null;
        double latitude = origin.getLatitude() + Math.toDegrees(northM / EARTH_RADIUS_M);
        double longitude = origin.getLongitude() + Math.toDegrees(
                eastM / (EARTH_RADIUS_M * Math.cos(Math.toRadians(origin.getLatitude()))));
        if (!Double.isFinite(latitude) || !Double.isFinite(longitude)
                || Math.abs(latitude) > 90 || Math.abs(longitude) > 180) return null;
        return new LatLng(latitude, longitude);
    }

    private static String summary(LoadedTrip trip) {
        if (trip.timeline.isEmpty()) {
            return "No replay positions were recorded. Sensor data may still be available in the CSV.";
        }
        if (trip.deadReckoningCount > 0) {
            return trip.rawGnssCount + " raw GNSS points · " + trip.deadReckoningCount
                    + " logged dead-reckoning points";
        }
        return trip.rawGnssCount + " recorded raw GNSS points"
                + (trip.legacyRawGnssOnly ? " · legacy GNSS replay" : "");
    }

    private void build(Bundle state) {
        LinearLayout root = AppUi.column(this);
        LinearLayout header = AppUi.column(this);
        header.setPadding(AppUi.dp(this, 20), AppUi.dp(this, 16), AppUi.dp(this, 20), 0);
        header.addView(AppUi.header(this, getString(R.string.ui_trip_replay), "Recorded sources · visual playback"));
        root.addView(header);

        FrameLayout mapContainer = new FrameLayout(this);
        mapView = new MapView(this);
        mapView.onCreate(state);
        mapContainer.addView(mapView, new FrameLayout.LayoutParams(-1, -1));
        root.addView(mapContainer, new LinearLayout.LayoutParams(-1, 0, 1));
        mapView.getMapAsync(loadedMap -> {
            map = loadedMap;
            map.setStyle("https://tiles.openfreemap.org/styles/dark", style -> {
                if (isDestroyed()) return;
                mapReady = true;
                render();
            });
        });

        LinearLayout controls = AppUi.card(this);
        LinearLayout sourceRow = AppUi.row(this);
        sourceRow.addView(AppUi.statusChip(this, "TRIP REPLAY", AppUi.GOLD));
        pointDetail = AppUi.text(this, "Loading points…", 12, AppUi.MUTED);
        pointDetail.setGravity(Gravity.END);
        sourceRow.addView(pointDetail, AppUi.weighted(this, 12));
        controls.addView(sourceRow);
        AppUi.gap(controls, 9);
        // Keep the source key in the native sheet instead of relying on a map
        // overlay, which some GPU/MapLibre surfaces can obscure.
        LinearLayout sourceKey = AppUi.row(this);
        sourceKey.setContentDescription("Trace key: blue is recorded raw GNSS; gold is logged dead-reckoning estimate");
        sourceKey.addView(AppUi.statusChip(this, "GNSS", GPS_ROUTE_COLOR));
        LinearLayout.LayoutParams idrKeyParams = new LinearLayout.LayoutParams(-2, -2);
        idrKeyParams.setMarginStart(AppUi.dp(this, 7));
        sourceKey.addView(AppUi.statusChip(this, "IDR", IDR_TRACK_COLOR), idrKeyParams);
        controls.addView(sourceKey);
        AppUi.gap(controls, 12);
        status = AppUi.text(this, getString(R.string.ui_loading_your_trip), 13, AppUi.MUTED);
        controls.addView(status);
        AppUi.gap(controls, 12);
        seek = new SeekBar(this);
        seek.setContentDescription("Replay position");
        seek.setEnabled(false);
        controls.addView(seek, new LinearLayout.LayoutParams(-1, AppUi.dp(this, 48)));
        seek.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override public void onStartTrackingTouch(SeekBar bar) { pause(); }
            @Override public void onStopTrackingTouch(SeekBar bar) { }
            @Override public void onProgressChanged(SeekBar bar, int value, boolean user) {
                if (user && vehicleMarker != null) {
                    index = value;
                    showPoint();
                }
            }
        });
        LinearLayout buttons = AppUi.row(this);
        play = AppUi.button(this, getString(R.string.ui_play_replay), true, () -> {
            if (playing) {
                pause();
            } else {
                if (index >= timeline.size() - 1) index = 0;
                playing = true;
                play.setText("Pause");
                handler.removeCallbacks(step);
                handler.post(step);
            }
        });
        play.setEnabled(false);
        buttons.addView(play, AppUi.weighted(this, 0));
        buttons.addView(AppUi.button(this, getString(R.string.ui_restart), false, () -> {
            pause();
            index = 0;
            showPoint();
        }), AppUi.weighted(this, 8));
        controls.addView(buttons);
        AppUi.gap(controls, 10);
        controls.addView(AppUi.text(this,
                "Playback timing is accelerated; map tiles and recorded data may differ from the original trip.",
                11, AppUi.MUTED));
        root.addView(controls);
        AppUi.install(this, root);
        mapView.addOnDidFailLoadingMapListener(error -> {
            if (!isDestroyed()) status.setText("Map could not load. Check your connection and reopen this trip.");
        });
    }

    private void render() {
        if (!loaded || !mapReady || drawn || timeline.isEmpty()) return;
        drawn = true;
        for (List<LatLng> segment : rawGnssSegments) {
            if (segment.size() > 1) {
                map.addPolyline(new PolylineOptions().addAll(segment).color(GPS_ROUTE_COLOR).width(5));
            }
        }
        for (List<LatLng> segment : deadReckoningSegments) {
            if (segment.size() > 1) {
                map.addPolyline(new PolylineOptions().addAll(segment).color(IDR_TRACK_COLOR).width(5));
            }
        }
        if (tripStart != null) {
            startMarker = map.addMarker(new MarkerOptions().position(tripStart).title("Trip start"));
        }
        if (tripEnd != null && !sameCoordinate(tripStart, tripEnd)) {
            endMarker = map.addMarker(new MarkerOptions().position(tripEnd).title("Trip end"));
        }
        ReplayPoint first = timeline.get(0);
        vehicleMarker = map.addMarker(new MarkerOptions().position(first.position).title(Brand.NAME + " replay position"));
        focusTrip(first.position);
        play.setEnabled(true);
        seek.setEnabled(true);
    }

    private void focusTrip(LatLng fallback) {
        double minLatitude = fallback.getLatitude();
        double maxLatitude = fallback.getLatitude();
        double minLongitude = fallback.getLongitude();
        double maxLongitude = fallback.getLongitude();
        LatLngBounds.Builder bounds = new LatLngBounds.Builder();
        for (ReplayPoint point : timeline) {
            bounds.include(point.position);
            minLatitude = Math.min(minLatitude, point.position.getLatitude());
            maxLatitude = Math.max(maxLatitude, point.position.getLatitude());
            minLongitude = Math.min(minLongitude, point.position.getLongitude());
            maxLongitude = Math.max(maxLongitude, point.position.getLongitude());
        }
        boolean hasSpan = Math.abs(maxLatitude - minLatitude) > 0.00001
                || Math.abs(maxLongitude - minLongitude) > 0.00001;
        if (hasSpan) {
            map.moveCamera(CameraUpdateFactory.newLatLngBounds(bounds.build(), AppUi.dp(this, 48)));
        } else {
            map.moveCamera(CameraUpdateFactory.newLatLngZoom(fallback, 16));
        }
    }

    private static boolean sameCoordinate(LatLng left, LatLng right) {
        return left != null && right != null
                && Math.abs(left.getLatitude() - right.getLatitude()) < 0.0000001
                && Math.abs(left.getLongitude() - right.getLongitude()) < 0.0000001;
    }

    private void showPoint() {
        if (vehicleMarker == null || timeline.isEmpty()) return;
        index = Math.max(0, Math.min(index, timeline.size() - 1));
        ReplayPoint point = timeline.get(index);
        vehicleMarker.setPosition(point.position);
        map.moveCamera(CameraUpdateFactory.newLatLng(point.position));
        seek.setProgress(index);
        status.setText("Point " + (index + 1) + " of " + timeline.size() + " · " + pointDescription(point.source));
        pointDetail.setText((index + 1) + " / " + timeline.size() + " points");
    }

    private static String pointDescription(Source source) {
        switch (source) {
            case RAW_GNSS:
                return "raw GNSS observation";
            case DEAD_RECKONING:
                return "dead-reckoning state estimate";
            default:
                return "logged filter state";
        }
    }

    private void pause() {
        playing = false;
        handler.removeCallbacks(step);
        if (play != null) play.setText("Play replay");
    }

    private final Runnable step = new Runnable() {
        @Override public void run() {
            if (!playing || vehicleMarker == null) return;
            showPoint();
            if (index >= timeline.size() - 1) {
                pause();
                return;
            }
            index = Math.min(timeline.size() - 1, index + Math.max(1, timeline.size() / 200));
            handler.postDelayed(this, 250);
        }
    };

    @Override protected void onStart() { super.onStart(); mapView.onStart(); }
    @Override protected void onResume() { super.onResume(); mapView.onResume(); }
    @Override protected void onPause() { pause(); mapView.onPause(); super.onPause(); }
    @Override protected void onStop() { mapView.onStop(); super.onStop(); }
    @Override protected void onSaveInstanceState(Bundle out) { super.onSaveInstanceState(out); mapView.onSaveInstanceState(out); }
    @Override public void onLowMemory() { super.onLowMemory(); mapView.onLowMemory(); }
    @Override protected void onDestroy() { handler.removeCallbacksAndMessages(null); mapView.onDestroy(); super.onDestroy(); }
}
