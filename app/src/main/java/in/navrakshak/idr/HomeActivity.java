package in.navrakshak.idr;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.hardware.Sensor;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.Environment;
import android.view.Gravity;
import android.view.View;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;

import java.io.File;

/**
 * Compact pre-drive command screen. The map artwork is intentionally neutral: it is a visual
 * entry point, not a claimed location or route before the user has begun a trip.
 */
public final class HomeActivity extends Activity {
    private TextView tripStatus;
    private AppUi.StatusChip readinessChip;
    private TextView readinessCopy;

    @Override public void onCreate(Bundle state) {
        androidx.core.splashscreen.SplashScreen splash = androidx.core.splashscreen.SplashScreen.installSplashScreen(this);
        super.onCreate(state);
        build();
        splash.setOnExitAnimationListener(provider -> provider.getView().animate()
                .alpha(0f)
                .setDuration(250)
                .withEndAction(provider::remove)
                .start());
    }

    private void open(Class<?> destination) {
        startActivity(new Intent(this, destination));
    }

    private void openSimulation() {
        Intent intent = new Intent(this, MainActivity.class);
        intent.putExtra("open_simulation", true);
        startActivity(intent);
    }

    private void build() {
        LinearLayout root = AppUi.page(this);

        // A small, recognisable app bar rather than a marketing hero.
        LinearLayout header = AppUi.row(this);
        header.addView(AppUi.logo(this, 42));
        LinearLayout wordmark = AppUi.column(this);
        TextView name = AppUi.title(this, Brand.NAME, 21);
        wordmark.addView(name);
        TextView caption = AppUi.text(this, "RELIABLE NAVIGATION BEYOND GPS", 9, AppUi.MUTED);
        caption.setLetterSpacing(.08f);
        LinearLayout.LayoutParams captionParams = new LinearLayout.LayoutParams(-2, -2);
        captionParams.topMargin = AppUi.dp(this, 3);
        wordmark.addView(caption, captionParams);
        header.addView(wordmark, AppUi.weighted(this, 11));
        readinessChip = AppUi.statusChip(this, "CHECKING", AppUi.AMBER);
        header.addView(readinessChip);
        root.addView(header);
        AppUi.gap(root, 17);

        // This deliberately has no dot, pin, route, or location label. Until a live trip starts,
        // the product should not imply that it knows where the user is.
        FrameLayout mapPreview = new FrameLayout(this);
        mapPreview.setBackground(AppUi.panel(this));
        mapPreview.setClipToOutline(true);
        mapPreview.setContentDescription("Neutral map preview. Start navigation to choose a destination.");
        MapPreviewView artwork = new MapPreviewView(this);
        mapPreview.addView(artwork, new FrameLayout.LayoutParams(-1, -1));

        TextView previewBadge = AppUi.text(this, "MAP PREVIEW", 10, AppUi.MUTED);
        previewBadge.setLetterSpacing(.1f);
        previewBadge.setTypeface(android.graphics.Typeface.create("sans-serif-medium", android.graphics.Typeface.NORMAL));
        previewBadge.setPadding(AppUi.dp(this, 10), AppUi.dp(this, 7), AppUi.dp(this, 10), AppUi.dp(this, 7));
        previewBadge.setBackground(AppUi.shape(this, 0xD9091220, 10));
        FrameLayout.LayoutParams previewBadgeParams = new FrameLayout.LayoutParams(-2, -2, Gravity.TOP | Gravity.START);
        previewBadgeParams.setMargins(AppUi.dp(this, 13), AppUi.dp(this, 13), 0, 0);
        mapPreview.addView(previewBadge, previewBadgeParams);

        TextView previewAction = AppUi.text(this, "Plan a route  ›", 13, AppUi.TEXT);
        previewAction.setTypeface(android.graphics.Typeface.create("sans-serif-medium", android.graphics.Typeface.NORMAL));
        previewAction.setPadding(AppUi.dp(this, 12), AppUi.dp(this, 9), AppUi.dp(this, 12), AppUi.dp(this, 9));
        previewAction.setBackground(AppUi.shape(this, 0xE4091220, 11));
        FrameLayout.LayoutParams previewActionParams = new FrameLayout.LayoutParams(-2, -2, Gravity.BOTTOM | Gravity.END);
        previewActionParams.setMargins(0, 0, AppUi.dp(this, 13), AppUi.dp(this, 13));
        mapPreview.addView(previewAction, previewActionParams);
        AppUi.tappable(mapPreview, () -> open(MainActivity.class));
        root.addView(mapPreview, new LinearLayout.LayoutParams(-1, AppUi.dp(this, 204)));

        AppUi.gap(root, 13);
        root.addView(AppUi.button(this, "Start navigation", true, () -> open(MainActivity.class)),
                new LinearLayout.LayoutParams(-1, -2));
        AppUi.gap(root, 11);

        LinearLayout readiness = AppUi.row(this);
        readiness.setGravity(Gravity.CENTER_VERTICAL);
        readiness.setPadding(AppUi.dp(this, 13), AppUi.dp(this, 11), AppUi.dp(this, 13), AppUi.dp(this, 11));
        readiness.setBackground(AppUi.outlinedShape(this, AppUi.SURFACE, 14, AppUi.BORDER));
        View readinessDot = new View(this);
        readinessDot.setBackground(AppUi.shape(this, AppUi.GREEN, 99));
        readiness.addView(readinessDot, new LinearLayout.LayoutParams(AppUi.dp(this, 7), AppUi.dp(this, 7)));
        readinessCopy = AppUi.text(this, "Checking phone readiness…", 12, AppUi.MUTED);
        readinessCopy.setMaxLines(2);
        LinearLayout.LayoutParams readinessCopyParams = new LinearLayout.LayoutParams(0, -2, 1);
        readinessCopyParams.setMarginStart(AppUi.dp(this, 9));
        readiness.addView(readinessCopy, readinessCopyParams);
        root.addView(readiness);

        AppUi.gap(root, 22);
        root.addView(AppUi.title(this, "Quick actions", 18));
        AppUi.gap(root, 11);

        LinearLayout firstRow = AppUi.row(this);
        firstRow.addView(quickAction(1, "Trips", "Saved routes", () -> open(LogsActivity.class)),
                new LinearLayout.LayoutParams(0, AppUi.dp(this, 112), 1));
        LinearLayout.LayoutParams secondTile = new LinearLayout.LayoutParams(0, AppUi.dp(this, 112), 1);
        secondTile.setMarginStart(AppUi.dp(this, 11));
        firstRow.addView(quickAction(2, "Outage test", "Simulate GNSS loss", this::openSimulation), secondTile);
        root.addView(firstRow);
        AppUi.gap(root, 11);

        LinearLayout secondRow = AppUi.row(this);
        secondRow.addView(quickAction(0, "Sensors", "Phone health", () -> open(SettingsActivity.class)),
                new LinearLayout.LayoutParams(0, AppUi.dp(this, 112), 1));
        LinearLayout.LayoutParams fourthTile = new LinearLayout.LayoutParams(0, AppUi.dp(this, 112), 1);
        fourthTile.setMarginStart(AppUi.dp(this, 11));
        secondRow.addView(quickAction(3, "Settings", "Navigation setup", () -> open(SettingsActivity.class)), fourthTile);
        root.addView(secondRow);

        AppUi.gap(root, 15);
        LinearLayout tripFooter = AppUi.row(this);
        tripFooter.setPadding(AppUi.dp(this, 2), 0, AppUi.dp(this, 2), 0);
        TextView localLabel = AppUi.text(this, "ON-DEVICE TRIPS", 10, AppUi.MUTED);
        localLabel.setLetterSpacing(.08f);
        tripFooter.addView(localLabel);
        tripStatus = AppUi.text(this, "", 12, AppUi.MUTED);
        tripStatus.setGravity(Gravity.END);
        tripFooter.addView(tripStatus, AppUi.weighted(this, 10));
        root.addView(tripFooter);
    }

    private View quickAction(int icon, String title, String description, Runnable action) {
        LinearLayout tile = AppUi.column(this);
        tile.setGravity(Gravity.CENTER_HORIZONTAL);
        tile.setPadding(AppUi.dp(this, 12), AppUi.dp(this, 12), AppUi.dp(this, 12), AppUi.dp(this, 10));
        tile.setBackground(AppUi.panel(this));
        DashboardIconView mark = new DashboardIconView(this, icon);
        tile.addView(mark, new LinearLayout.LayoutParams(AppUi.dp(this, 34), AppUi.dp(this, 34)));
        AppUi.gap(tile, 8);
        TextView label = AppUi.title(this, title, 14);
        label.setGravity(Gravity.CENTER);
        tile.addView(label, new LinearLayout.LayoutParams(-1, -2));
        TextView detail = AppUi.text(this, description, 10, AppUi.MUTED);
        detail.setGravity(Gravity.CENTER);
        LinearLayout.LayoutParams detailParams = new LinearLayout.LayoutParams(-1, -2);
        detailParams.topMargin = AppUi.dp(this, 3);
        tile.addView(detail, detailParams);
        AppUi.tappable(tile, action);
        tile.setContentDescription(title + ". " + description);
        return tile;
    }

    @Override protected void onResume() {
        super.onResume();
        File dir = getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS);
        File[] files = dir == null ? null : dir.listFiles((directory, name) -> name.endsWith(".csv"));
        int count = files == null ? 0 : files.length;
        if (tripStatus != null) {
            tripStatus.setText(count == 0 ? "No trips recorded" : count + (count == 1 ? " trip recorded" : " trips recorded"));
        }
        updateReadiness();
    }

    private void updateReadiness() {
        if (readinessChip == null || readinessCopy == null) return;
        boolean locationAllowed = checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED;
        SensorManager manager = (SensorManager) getSystemService(SENSOR_SERVICE);
        boolean sensorsReady = manager != null
                && manager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER) != null
                && manager.getDefaultSensor(Sensor.TYPE_GYROSCOPE) != null
                && manager.getDefaultSensor(Sensor.TYPE_GRAVITY) != null
                && manager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR) != null;
        if (!locationAllowed) {
            readinessChip.setState("LOCATION SETUP", AppUi.AMBER);
            readinessCopy.setText("Allow precise location, then begin outdoors for a fresh GNSS fix.");
        } else if (!sensorsReady) {
            readinessChip.setState("SENSOR CHECK", AppUi.AMBER);
            readinessCopy.setText("Some motion sensors are unavailable. Review phone health before a trip.");
        } else {
            readinessChip.setState("PHONE READY", AppUi.GREEN);
            readinessCopy.setText("Mount securely and begin outdoors for a fresh GNSS fix.");
        }
    }

    /** Original decorative road network used only before live map data has been requested. */
    private static final class MapPreviewView extends View {
        private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final Path path = new Path();

        MapPreviewView(android.content.Context context) {
            super(context);
        }

        @Override protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);
            float width = getWidth();
            float height = getHeight();
            canvas.drawColor(Color.rgb(14, 29, 47));

            // City blocks and parks sit under the roads to give the preview depth without
            // presenting any actual user's position, destination, or street names.
            paint.setStyle(Paint.Style.FILL);
            paint.setColor(Color.rgb(20, 42, 62));
            for (int row = 0; row < 4; row++) {
                for (int column = 0; column < 5; column++) {
                    float left = width * (.03f + column * .21f) + (row % 2) * width * .018f;
                    float top = height * (.16f + row * .20f);
                    canvas.drawRoundRect(left, top, left + width * .16f, top + height * .12f,
                            8, 8, paint);
                }
            }
            paint.setColor(Color.rgb(20, 63, 60));
            canvas.drawRoundRect(width * .05f, height * .30f, width * .22f, height * .57f, 16, 16, paint);
            canvas.drawRoundRect(width * .74f, height * .14f, width * .94f, height * .31f, 16, 16, paint);

            road(canvas, new float[]{-.04f, .77f, .20f, .65f, .38f, .52f, .56f, .44f, .78f, .27f, 1.05f, .18f}, width, height, 21, Color.rgb(7, 16, 28));
            road(canvas, new float[]{-.04f, .77f, .20f, .65f, .38f, .52f, .56f, .44f, .78f, .27f, 1.05f, .18f}, width, height, 15, Color.rgb(63, 88, 115));
            road(canvas, new float[]{.13f, -.08f, .22f, .18f, .36f, .37f, .49f, .58f, .58f, 1.08f}, width, height, 16, Color.rgb(7, 16, 28));
            road(canvas, new float[]{.13f, -.08f, .22f, .18f, .36f, .37f, .49f, .58f, .58f, 1.08f}, width, height, 10, Color.rgb(54, 79, 104));
            road(canvas, new float[]{.64f, -.04f, .60f, .20f, .65f, .40f, .73f, .58f, .92f, .72f, 1.06f, .93f}, width, height, 13, Color.rgb(7, 16, 28));
            road(canvas, new float[]{.64f, -.04f, .60f, .20f, .65f, .40f, .73f, .58f, .92f, .72f, 1.06f, .93f}, width, height, 8, Color.rgb(54, 79, 104));

            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(Math.max(1f, width * .004f));
            paint.setColor(0x607FA2C6);
            for (int i = 0; i < 4; i++) {
                float offset = height * (.18f + i * .18f);
                canvas.drawLine(0, offset, width, offset - height * .12f, paint);
            }
        }

        private void road(Canvas canvas, float[] points, float width, float height, float stroke, int color) {
            path.reset();
            path.moveTo(points[0] * width, points[1] * height);
            for (int i = 2; i < points.length; i += 2) {
                path.lineTo(points[i] * width, points[i + 1] * height);
            }
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeCap(Paint.Cap.ROUND);
            paint.setStrokeJoin(Paint.Join.ROUND);
            paint.setStrokeWidth(Math.max(2f, width * stroke / 390f));
            paint.setColor(color);
            canvas.drawPath(path, paint);
        }
    }
}
