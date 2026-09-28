package in.navrakshak.idr;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;
import android.os.Bundle;
import android.os.Environment;
import android.view.Gravity;
import android.view.View;
import android.widget.LinearLayout;

import java.io.File;
import java.text.DateFormat;
import java.util.Arrays;
import java.util.Date;

/** Local, deliberately compact trip library. Files leave the device only through share(). */
public final class LogsActivity extends Activity {
    @Override public void onCreate(Bundle state) { super.onCreate(state); }

    @Override protected void onResume() {
        super.onResume();
        build();
    }

    private void build() {
        LinearLayout root = AppUi.page(this);
        root.addView(header());

        File dir = getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS);
        File[] files = dir == null ? null : dir.listFiles((directory, name) -> name.endsWith(".csv"));
        if (files == null || files.length == 0) {
            empty(root);
            return;
        }

        Arrays.sort(files, (a, b) -> Long.compare(b.lastModified(), a.lastModified()));
        LinearLayout meta = AppUi.row(this);
        meta.addView(AppUi.statusChip(this, files.length + (files.length == 1 ? " TRIP" : " TRIPS"), AppUi.GREEN));
        meta.addView(new View(this), new LinearLayout.LayoutParams(0, 1, 1));
        meta.addView(AppUi.text(this, "Stored on this device", 12, AppUi.MUTED));
        root.addView(meta);
        AppUi.gap(root, 18);
        root.addView(AppUi.overline(this, "RECENT TRIPS"));
        AppUi.gap(root, 9);

        for (int i = 0; i < files.length; i++) {
            root.addView(tripRow(files[i], i));
            if (i < files.length - 1) divider(root);
        }
        AppUi.gap(root, 16);
        root.addView(AppUi.text(this, "Replay separates recorded GNSS and logged IDR estimates; neither is ground truth.", 12, AppUi.MUTED));
    }

    private LinearLayout header() {
        LinearLayout header = AppUi.row(this);
        header.setGravity(Gravity.CENTER_VERTICAL);
        header.addView(AppUi.iconButton(this, R.drawable.ic_arrow_back, "Go back", this::finish), new LinearLayout.LayoutParams(AppUi.dp(this, 48), AppUi.dp(this, 48)));
        LinearLayout words = AppUi.column(this);
        words.addView(AppUi.title(this, "Trips", 22));
        AppUi.gap(words, 3);
        words.addView(AppUi.text(this, "Recorded locally", 12, AppUi.MUTED));
        header.addView(words, AppUi.weighted(this, 12));
        header.addView(AppUi.statusChip(this, "PRIVATE", AppUi.GREEN));
        return header;
    }

    private void empty(LinearLayout root) {
        AppUi.gap(root, 40);
        LinearLayout card = AppUi.card(this);
        card.setGravity(Gravity.CENTER_HORIZONTAL);
        card.addView(new RouteThumbnail(this, 0), new LinearLayout.LayoutParams(AppUi.dp(this, 92), AppUi.dp(this, 92)));
        AppUi.gap(card, 18);
        card.addView(AppUi.title(this, "No trips yet", 23));
        AppUi.gap(card, 7);
        android.widget.TextView copy = AppUi.text(this, "Start navigation to record a local sensor and position log.", 14, AppUi.MUTED);
        copy.setGravity(Gravity.CENTER);
        card.addView(copy);
        AppUi.gap(card, 22);
        card.addView(AppUi.button(this, "Start navigation", true, () -> startActivity(new Intent(this, MainActivity.class))), new LinearLayout.LayoutParams(-1, -2));
        root.addView(card);
    }

    private LinearLayout tripRow(File file, int variation) {
        LinearLayout row = AppUi.row(this);
        row.setPadding(0, AppUi.dp(this, 11), 0, AppUi.dp(this, 11));
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.addView(new RouteThumbnail(this, variation), new LinearLayout.LayoutParams(AppUi.dp(this, 68), AppUi.dp(this, 68)));

        LinearLayout words = AppUi.column(this);
        words.addView(AppUi.title(this, DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT).format(new Date(file.lastModified())), 15));
        AppUi.gap(words, 4);
        words.addView(AppUi.text(this, displayName(file), 12, AppUi.MUTED));
        AppUi.gap(words, 6);
        words.addView(AppUi.text(this, "CSV log  ·  " + android.text.format.Formatter.formatShortFileSize(this, file.length()), 12, AppUi.GOLD));
        row.addView(words, AppUi.weighted(this, 12));

        LinearLayout actions = AppUi.column(this);
        actions.setGravity(Gravity.CENTER);
        actions.addView(AppUi.iconButton(this, R.drawable.ic_play, "Open replay", () -> replay(file)), new LinearLayout.LayoutParams(AppUi.dp(this, 44), AppUi.dp(this, 44)));
        AppUi.gap(actions, 5);
        actions.addView(AppUi.iconButton(this, R.drawable.ic_share, "Share CSV", () -> share(file)), new LinearLayout.LayoutParams(AppUi.dp(this, 44), AppUi.dp(this, 44)));
        row.addView(actions);
        AppUi.tappable(row, () -> replay(file));
        return row;
    }

    private void replay(File file) {
        Intent intent = new Intent(this, ReplayActivity.class);
        intent.putExtra("csv_path", file.getAbsolutePath());
        startActivity(intent);
    }

    private void divider(LinearLayout parent) {
        View line = new View(this);
        line.setBackgroundColor(AppUi.BORDER);
        parent.addView(line, new LinearLayout.LayoutParams(-1, AppUi.dp(this, 1)));
    }

    private String displayName(File file) {
        String name = file.getName();
        if (name.startsWith("navrakshak_") || name.startsWith(Brand.LOG_PREFIX)) return Brand.NAME + " trip log";
        return name;
    }

    private void share(File file) {
        try {
            Intent intent = new Intent(Intent.ACTION_SEND);
            intent.setType("text/csv");
            android.net.Uri uri = androidx.core.content.FileProvider.getUriForFile(this, getPackageName() + ".files", file);
            intent.putExtra(Intent.EXTRA_STREAM, uri);
            intent.setClipData(android.content.ClipData.newRawUri("Path Sārathi trip CSV", uri));
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            startActivity(Intent.createChooser(intent, "Share trip log"));
        } catch (Exception error) {
            new android.app.AlertDialog.Builder(this).setTitle("Unable to share").setMessage("This trip file is no longer available, or no sharing app is installed.").setPositiveButton("OK", null).show();
        }
    }

    /** A visual index only; it never claims to be the recorded route. */
    private static final class RouteThumbnail extends View {
        private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final int variation;
        RouteThumbnail(android.content.Context context, int variation) { super(context); this.variation = variation; }
        @Override protected void onDraw(Canvas canvas) {
            int w = getWidth(), h = getHeight();
            paint.setStyle(Paint.Style.FILL); paint.setColor(AppUi.PANEL);
            canvas.drawRoundRect(0, 0, w, h, AppUi.dp(getContext(), 14), AppUi.dp(getContext(), 14), paint);
            paint.setStyle(Paint.Style.STROKE); paint.setStrokeWidth(AppUi.dp(getContext(), 1)); paint.setColor(AppUi.BORDER);
            for (int i = -h; i < w + h; i += AppUi.dp(getContext(), 16)) canvas.drawLine(i, 0, i + h, h, paint);
            for (int i = 0; i < h; i += AppUi.dp(getContext(), 19)) canvas.drawLine(0, i, w, i + w / 3f, paint);
            Path route = new Path();
            if (variation % 3 == 0) { route.moveTo(w * .17f, h * .78f); route.cubicTo(w*.30f,h*.55f,w*.45f,h*.72f,w*.57f,h*.42f); route.lineTo(w*.82f,h*.20f); }
            else if (variation % 3 == 1) { route.moveTo(w*.18f,h*.25f); route.lineTo(w*.43f,h*.38f); route.cubicTo(w*.52f,h*.48f,w*.39f,h*.67f,w*.75f,h*.78f); }
            else { route.moveTo(w*.15f,h*.72f); route.lineTo(w*.35f,h*.60f); route.lineTo(w*.45f,h*.26f); route.cubicTo(w*.57f,h*.46f,w*.67f,h*.45f,w*.82f,h*.18f); }
            paint.setStyle(Paint.Style.STROKE); paint.setStrokeWidth(AppUi.dp(getContext(), 4)); paint.setStrokeCap(Paint.Cap.ROUND); paint.setColor(AppUi.GOLD); canvas.drawPath(route, paint);
            paint.setStyle(Paint.Style.FILL); paint.setColor(AppUi.TEXT); canvas.drawCircle(w*.17f, variation % 3 == 1 ? h*.25f : h*.78f, AppUi.dp(getContext(), 4), paint);
            paint.setColor(AppUi.GOLD); canvas.drawCircle(w*.82f, variation % 3 == 1 ? h*.78f : h*.20f, AppUi.dp(getContext(), 4), paint);
        }
    }
}
