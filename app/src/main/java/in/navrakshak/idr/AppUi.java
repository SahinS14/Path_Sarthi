package in.navrakshak.idr;

import android.app.Activity;
import android.content.Context;
import android.content.res.ColorStateList;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.RippleDrawable;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowCompat;
import androidx.core.view.WindowInsetsCompat;

import java.io.InputStream;

/** Shared native design system. Dimensions are dp and text sizes are sp. */
final class AppUi {
    static final int BG = Color.rgb(9, 18, 32);
    static final int SURFACE = Color.rgb(13, 26, 43);
    static final int PANEL = Color.rgb(19, 35, 55);
    static final int ELEVATED = Color.rgb(27, 46, 69);
    static final int BORDER = Color.rgb(49, 70, 94);
    static final int TEXT = Color.rgb(243, 247, 252);
    static final int MUTED = Color.rgb(174, 190, 211);
    static final int GOLD = Color.rgb(235, 195, 110);
    static final int GOLD_DARK = Color.rgb(159, 119, 47);
    static final int GREEN = Color.rgb(111, 221, 182);
    static final int AMBER = Color.rgb(255, 199, 117);
    static final int DANGER = Color.rgb(250, 142, 142);

    private static Bitmap brand;

    static int dp(Context c, float value) {
        return Math.round(value * c.getResources().getDisplayMetrics().density);
    }

    static GradientDrawable shape(Context c, int color, int radius) {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(color);
        drawable.setCornerRadius(dp(c, radius));
        return drawable;
    }

    static GradientDrawable outlinedShape(Context c, int color, int radius, int strokeColor) {
        GradientDrawable drawable = shape(c, color, radius);
        drawable.setStroke(dp(c, 1), strokeColor);
        return drawable;
    }

    static GradientDrawable panel(Context c) {
        return outlinedShape(c, PANEL, 20, BORDER);
    }

    static GradientDrawable topSheet(Context c) {
        GradientDrawable drawable = outlinedShape(c, SURFACE, 24, BORDER);
        drawable.setCornerRadii(new float[]{dp(c, 24), dp(c, 24), dp(c, 24), dp(c, 24), 0, 0, 0, 0});
        return drawable;
    }

    static LinearLayout column(Context c) {
        LinearLayout view = new LinearLayout(c);
        view.setOrientation(LinearLayout.VERTICAL);
        return view;
    }

    static LinearLayout row(Context c) {
        LinearLayout view = new LinearLayout(c);
        view.setGravity(Gravity.CENTER_VERTICAL);
        return view;
    }

    static TextView text(Context c, String value, int size, int color) {
        TextView view = new TextView(c);
        view.setText(value);
        view.setTextSize(size);
        view.setTextColor(color);
        view.setIncludeFontPadding(false);
        view.setFontFeatureSettings("kern");
        view.setLineSpacing(dp(c, 3), 1f);
        return view;
    }

    static TextView title(Context c, String value, int size) {
        TextView view = text(c, value, size, TEXT);
        view.setTypeface(Typeface.create("sans-serif", Typeface.BOLD));
        return view;
    }

    static TextView overline(Context c, String value) {
        TextView view = text(c, value.toUpperCase(), 11, GOLD);
        view.setTypeface(Typeface.create("sans-serif-medium", Typeface.NORMAL));
        view.setLetterSpacing(.12f);
        return view;
    }

    static void gap(LinearLayout parent, int height) {
        View space = new View(parent.getContext());
        parent.addView(space, new LinearLayout.LayoutParams(1, dp(parent.getContext(), height)));
    }

    static LinearLayout card(Context c) {
        LinearLayout view = column(c);
        view.setPadding(dp(c, 18), dp(c, 18), dp(c, 18), dp(c, 18));
        view.setBackground(panel(c));
        view.setClipToOutline(false);
        ViewCompat.setElevation(view, dp(c, 2));
        return view;
    }

    static LinearLayout.LayoutParams weighted(Context c, int marginStart) {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1);
        params.setMarginStart(dp(c, marginStart));
        return params;
    }

    static void tappable(View view, Runnable action) {
        view.setClickable(true);
        view.setFocusable(true);
        view.setOnClickListener(v -> action.run());
        view.setForeground(new RippleDrawable(ColorStateList.valueOf(0x20FFFFFF), null, null));
    }

    static void style(android.widget.Button button, boolean primary) {
        Context context = button.getContext();
        button.setAllCaps(false);
        button.setTextSize(14);
        button.setTypeface(Typeface.create("sans-serif-medium", Typeface.NORMAL));
        button.setTextColor(new ColorStateList(
                new int[][]{new int[]{-android.R.attr.state_enabled}, new int[]{}},
                new int[]{MUTED, primary ? BG : TEXT}));
        button.setMinHeight(dp(context, 50));
        button.setMinimumHeight(dp(context, 50));
        button.setPadding(dp(context, 16), dp(context, 10), dp(context, 16), dp(context, 10));
        GradientDrawable normal = primary ? shape(context, GOLD, 14) : outlinedShape(context, ELEVATED, 14, BORDER);
        GradientDrawable pressed = primary ? shape(context, Color.rgb(255, 214, 128), 14) : outlinedShape(context, Color.rgb(40, 61, 86), 14, GOLD_DARK);
        GradientDrawable disabled = shape(context, BORDER, 14);
        android.graphics.drawable.StateListDrawable states = new android.graphics.drawable.StateListDrawable();
        states.addState(new int[]{-android.R.attr.state_enabled}, disabled);
        states.addState(new int[]{android.R.attr.state_pressed}, pressed);
        states.addState(new int[]{}, normal);
        button.setBackground(new RippleDrawable(ColorStateList.valueOf(0x25FFFFFF), states, null));
    }

    static android.widget.Button button(Context c, String label, boolean primary, Runnable action) {
        android.widget.Button button = new android.widget.Button(c);
        button.setText(label);
        style(button, primary);
        button.setOnClickListener(v -> action.run());
        return button;
    }

    static ImageButton iconButton(Context c, int icon, String description, Runnable action) {
        ImageButton button = new ImageButton(c);
        button.setImageResource(icon);
        button.setImageTintList(ColorStateList.valueOf(TEXT));
        button.setContentDescription(description);
        button.setPadding(dp(c, 12), dp(c, 12), dp(c, 12), dp(c, 12));
        GradientDrawable normal = outlinedShape(c, ELEVATED, 14, BORDER);
        GradientDrawable pressed = outlinedShape(c, Color.rgb(40, 61, 86), 14, GOLD_DARK);
        android.graphics.drawable.StateListDrawable states = new android.graphics.drawable.StateListDrawable();
        states.addState(new int[]{android.R.attr.state_pressed}, pressed);
        states.addState(new int[]{}, normal);
        button.setBackground(new RippleDrawable(ColorStateList.valueOf(0x25FFFFFF), states, null));
        button.setOnClickListener(v -> action.run());
        return button;
    }

    static TextView badge(Context c, String label) {
        TextView view = text(c, label, 11, GOLD);
        view.setTypeface(Typeface.create("sans-serif-medium", Typeface.NORMAL));
        view.setPadding(dp(c, 10), dp(c, 6), dp(c, 10), dp(c, 6));
        view.setBackground(shape(c, 0xFF293A50, 9));
        return view;
    }

    static final class StatusChip extends LinearLayout {
        private final View dot;
        private final TextView label;

        StatusChip(Context c, String initialLabel, int initialColor) {
            super(c);
            setOrientation(HORIZONTAL);
            setGravity(Gravity.CENTER_VERTICAL);
            setPadding(dp(c, 10), dp(c, 7), dp(c, 11), dp(c, 7));
            setBackground(outlinedShape(c, ELEVATED, 999, BORDER));
            dot = new View(c);
            addView(dot, new LinearLayout.LayoutParams(dp(c, 7), dp(c, 7)));
            label = text(c, "", 11, TEXT);
            label.setTypeface(Typeface.create("sans-serif-medium", Typeface.NORMAL));
            LinearLayout.LayoutParams labelParams = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
            labelParams.setMarginStart(dp(c, 7));
            addView(label, labelParams);
            setState(initialLabel, initialColor);
        }

        void setState(String value, int color) {
            label.setText(value);
            dot.setBackground(shape(getContext(), color, 999));
        }
    }

    static StatusChip statusChip(Context c, String label, int color) {
        return new StatusChip(c, label, color);
    }

    static ImageView logo(Context c, int size) {
        ImageView image = new ImageView(c);
        if (brand == null) {
            try (InputStream input = c.getAssets().open("logo.png")) {
                BitmapFactory.Options options = new BitmapFactory.Options();
                options.inSampleSize = 4;
                brand = BitmapFactory.decodeStream(input, null, options);
            } catch (Exception ignored) {
                // The generated launcher resource still supplies the splash icon if this is unavailable.
            }
        }
        image.setImageBitmap(brand);
        image.setScaleType(ImageView.ScaleType.FIT_CENTER);
        image.setBackground(shape(c, Color.WHITE, 16));
        image.setClipToOutline(true);
        image.setContentDescription("Path Sārathi logo");
        image.setLayoutParams(new LinearLayout.LayoutParams(dp(c, size), dp(c, size)));
        return image;
    }

    static LinearLayout header(Activity activity, String title, String caption) {
        LinearLayout header = column(activity);
        LinearLayout top = row(activity);
        ImageButton back = iconButton(activity, R.drawable.ic_arrow_back, "Go back", activity::finish);
        top.addView(back, new LinearLayout.LayoutParams(dp(activity, 48), dp(activity, 48)));
        TextView heading = title(activity, title, 22);
        top.addView(heading, weighted(activity, 14));
        header.addView(top);
        if (caption != null) {
            gap(header, 12);
            header.addView(text(activity, caption, 13, MUTED));
        }
        gap(header, 22);
        return header;
    }

    static LinearLayout page(Activity activity) {
        ScrollView scroll = new ScrollView(activity);
        scroll.setFillViewport(true);
        scroll.setClipToPadding(false);
        LinearLayout content = column(activity);
        content.setPadding(dp(activity, 20), dp(activity, 20), dp(activity, 20), dp(activity, 32));
        scroll.addView(content);
        install(activity, scroll);
        return content;
    }

    static void install(Activity activity, View view) {
        view.setBackgroundColor(BG);
        WindowCompat.setDecorFitsSystemWindows(activity.getWindow(), false);
        FrameLayout safe = new FrameLayout(activity);
        safe.setBackgroundColor(BG);
        safe.addView(view, new FrameLayout.LayoutParams(-1, -1));
        ViewCompat.setOnApplyWindowInsetsListener(safe, (v, insets) -> {
            Insets bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() | WindowInsetsCompat.Type.displayCutout() | WindowInsetsCompat.Type.ime());
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom);
            return insets;
        });
        activity.setContentView(safe);
        WindowCompat.getInsetsController(activity.getWindow(), safe).setAppearanceLightStatusBars(false);
        WindowCompat.getInsetsController(activity.getWindow(), safe).setAppearanceLightNavigationBars(false);
        ViewCompat.requestApplyInsets(safe);
    }

    /** Keep the existing store name so prior user preferences survive the rebrand. */
    static android.content.SharedPreferences prefs(Context c) {
        return c.getSharedPreferences("navrakshak_preferences", Context.MODE_PRIVATE);
    }

    static final class StatusPanel extends ScrollView {
        StatusPanel(Context c) {
            super(c);
        }

        @Override protected void onMeasure(int width, int height) {
            if (MeasureSpec.getMode(height) == MeasureSpec.EXACTLY) {
                super.onMeasure(width, height);
                return;
            }
            int cap = dp(getContext(), 300);
            if (MeasureSpec.getMode(height) != MeasureSpec.UNSPECIFIED) {
                cap = Math.min(cap, MeasureSpec.getSize(height));
            }
            super.onMeasure(width, MeasureSpec.makeMeasureSpec(cap, MeasureSpec.AT_MOST));
        }
    }
}
