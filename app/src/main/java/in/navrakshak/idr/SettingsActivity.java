package in.navrakshak.idr;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.ColorStateList;
import android.hardware.Sensor;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.provider.Settings;
import android.view.Gravity;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.Switch;
import android.widget.TextView;

/** Short-form settings screen: real settings plus transparent device readiness. */
public final class SettingsActivity extends Activity {
    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        build();
    }

    private void build() {
        LinearLayout root = AppUi.page(this);
        root.addView(header());
        AppUi.gap(root, 20);

        section(root, "NAVIGATION");
        infoRow(root, 0, "Offline maps", "Map tiles and destination search need internet", null);
        divider(root);
        toggleRow(root, 0, "Spoken directions", "Voice guidance during an active trip", "voice", true);
        divider(root);
        toggleRow(root, 2, "Keep screen awake", "Keep the map visible while recording", "keep_awake", true);
        AppUi.gap(root, 20);

        section(root, "SYSTEM");
        boolean fineLocation = checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED;
        infoRow(root, 0, "GNSS assistance", fineLocation ? "Precise location permission granted" : "Location permission needs attention", this::openAppSettings);
        divider(root);
        SensorManager manager = (SensorManager) getSystemService(SENSOR_SERVICE);
        boolean sensors = has(manager, Sensor.TYPE_ACCELEROMETER) && has(manager, Sensor.TYPE_GYROSCOPE);
        infoRow(root, 3, "Sensor access", sensors ? "Accelerometer and gyroscope available" : "A required motion sensor is unavailable", this::openAppSettings);
        divider(root);
        infoRow(root, 1, "Trip library", "Saved locally in CSV format", () -> startActivity(new Intent(this, LogsActivity.class)));
        AppUi.gap(root, 20);

        section(root, "DIAGNOSTICS");
        diagnosticRow(root, 0, "Precise location", fineLocation, "Origin and GNSS corrections");
        divider(root);
        diagnosticRow(root, 3, "Accelerometer", has(manager, Sensor.TYPE_ACCELEROMETER), "Motion input");
        divider(root);
        diagnosticRow(root, 3, "Gyroscope", has(manager, Sensor.TYPE_GYROSCOPE), "Turn-rate input");
        divider(root);
        diagnosticRow(root, 0, "Mount alignment", has(manager, Sensor.TYPE_GRAVITY) && has(manager, Sensor.TYPE_ROTATION_VECTOR), "Gravity and orientation sensors");
        AppUi.gap(root, 20);

        section(root, "ABOUT");
        infoRow(root, 1, Brand.PRODUCT, "On-device GNSS + IMU navigation research", null);
        divider(root);
        infoRow(root, 1, "App version", "0.1.0 development build", null);
        AppUi.gap(root, 12);
        root.addView(AppUi.text(this, "During a GNSS interruption, shown positions are estimates and can drift.", 12, AppUi.MUTED));
    }

    private LinearLayout header() {
        LinearLayout row = AppUi.row(this);
        row.addView(AppUi.iconButton(this, R.drawable.ic_arrow_back, "Go back", this::finish), new LinearLayout.LayoutParams(AppUi.dp(this, 48), AppUi.dp(this, 48)));
        LinearLayout words = AppUi.column(this);
        words.addView(AppUi.title(this, "Settings", 22));
        AppUi.gap(words, 3);
        words.addView(AppUi.text(this, "Navigation, device, and privacy", 12, AppUi.MUTED));
        row.addView(words, AppUi.weighted(this, 12));
        row.addView(AppUi.statusChip(this, "ON DEVICE", AppUi.GREEN));
        return row;
    }

    private void section(LinearLayout root, String title) { root.addView(AppUi.overline(this, title)); AppUi.gap(root, 7); }

    private boolean has(SensorManager manager, int type) { return manager != null && manager.getDefaultSensor(type) != null; }

    private void toggleRow(LinearLayout parent, int icon, String title, String subtitle, String key, boolean fallback) {
        LinearLayout row = baseRow(icon, title, subtitle);
        Switch control = new Switch(this);
        control.setContentDescription(title);
        control.setShowText(false);
        control.setChecked(AppUi.prefs(this).getBoolean(key, fallback));
        control.setButtonTintList(new ColorStateList(new int[][]{new int[]{android.R.attr.state_checked}, new int[]{}}, new int[]{AppUi.GOLD, AppUi.MUTED}));
        control.setOnCheckedChangeListener((view, checked) -> AppUi.prefs(this).edit().putBoolean(key, checked).apply());
        row.addView(control);
        parent.addView(row);
    }

    private void infoRow(LinearLayout parent, int icon, String title, String subtitle, Runnable action) {
        LinearLayout row = baseRow(icon, title, subtitle);
        TextView tail = AppUi.text(this, action == null ? "" : "›", 28, AppUi.MUTED);
        tail.setGravity(Gravity.CENTER);
        row.addView(tail, new LinearLayout.LayoutParams(AppUi.dp(this, 30), AppUi.dp(this, 48)));
        if (action != null) AppUi.tappable(row, action);
        parent.addView(row);
    }

    private void diagnosticRow(LinearLayout parent, int icon, String title, boolean available, String subtitle) {
        LinearLayout row = baseRow(icon, title, subtitle);
        row.addView(AppUi.statusChip(this, available ? "READY" : "REVIEW", available ? AppUi.GREEN : AppUi.AMBER));
        parent.addView(row);
    }

    private LinearLayout baseRow(int icon, String title, String subtitle) {
        LinearLayout row = AppUi.row(this);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setMinimumHeight(AppUi.dp(this, 64));
        DashboardIconView glyph = new DashboardIconView(this, icon);
        row.addView(glyph, new LinearLayout.LayoutParams(AppUi.dp(this, 38), AppUi.dp(this, 48)));
        LinearLayout words = AppUi.column(this);
        words.addView(AppUi.text(this, title, 15, AppUi.TEXT));
        AppUi.gap(words, 4);
        words.addView(AppUi.text(this, subtitle, 12, AppUi.MUTED));
        row.addView(words, AppUi.weighted(this, 10));
        return row;
    }

    private void divider(LinearLayout parent) {
        View line = new View(this);
        line.setBackgroundColor(AppUi.BORDER);
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(-1, AppUi.dp(this, 1));
        params.setMargins(AppUi.dp(this, 48), 0, 0, 0);
        parent.addView(line, params);
    }

    private void openAppSettings() { startActivity(new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, android.net.Uri.parse("package:" + getPackageName()))); }
}
