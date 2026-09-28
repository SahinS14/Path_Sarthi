package in.navrakshak.idr;

import android.content.Context;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

/** Small OSM road graph stored inside the APK; it never needs network access. */
final class OfflineRoadMap {
    private static final double R = 6371000.0;
    private final List<double[][]> geoWays = new ArrayList<>();
    private final Map<String, List<String>> edges = new HashMap<>();
    private final Map<String, double[]> geoNodes = new HashMap<>();
    private double lat0, lon0; private boolean ready;

    OfflineRoadMap(Context context) {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(context.getAssets().open("demo_roads.osm.json")))) {
            StringBuilder text = new StringBuilder(); String line;
            while ((line = reader.readLine()) != null) text.append(line);
            JSONArray ways = new JSONObject(text.toString()).getJSONArray("elements");
            for (int i = 0; i < ways.length(); i++) {
                JSONArray geometry = ways.getJSONObject(i).optJSONArray("geometry"); if (geometry == null || geometry.length() < 2) continue;
                double[][] way = new double[geometry.length()][2];
                for (int j = 0; j < geometry.length(); j++) { JSONObject point = geometry.getJSONObject(j); way[j][0] = point.getDouble("lat"); way[j][1] = point.getDouble("lon"); }
                geoWays.add(way);
                for (int j = 0; j < way.length - 1; j++) connect(key(way[j]), way[j], key(way[j + 1]), way[j + 1]);
            }
        } catch (Exception ignored) { }
    }
    void setOrigin(double latitude, double longitude) { lat0 = latitude; lon0 = longitude; ready = true; }
    boolean isReady() { return ready && !geoWays.isEmpty(); }
    /** A local road observation; confidence is intentionally low for parallel/ambiguous roads. */
    static final class Match { final double east,north,heading,confidence,distanceM; Match(double e,double n,double h,double c,double d){east=e;north=n;heading=h;confidence=c;distanceM=d;} }
    double[] geo(double east, double north) { return new double[]{lat0 + Math.toDegrees(north / R), lon0 + Math.toDegrees(east / (R * Math.cos(Math.toRadians(lat0))))}; }
    List<double[][]> roads() { List<double[][]> local = new ArrayList<>(); if (!isReady()) return local; for (double[][] way : geoWays) { double[][] converted = new double[way.length][2]; for (int i=0;i<way.length;i++) converted[i] = xy(way[i]); local.add(converted); } return local; }
    List<double[]> route(double startE, double startN, double endE, double endN) {
        if (!isReady()) return Collections.emptyList(); String start = closest(startE, startN), end = closest(endE, endN); if (start == null || end == null) return Collections.emptyList();
        Map<String, Double> cost = new HashMap<>(); Map<String, String> prev = new HashMap<>(); PriorityQueue<Entry> queue = new PriorityQueue<>(); cost.put(start, 0.0); queue.add(new Entry(start, 0)); Set<String> done = new HashSet<>();
        while (!queue.isEmpty()) { Entry current = queue.poll(); if (!done.add(current.node)) continue; if (current.node.equals(end)) break; for (String next : edges.getOrDefault(current.node, Collections.emptyList())) { double candidate = current.cost + distance(xy(geoNodes.get(current.node)), xy(geoNodes.get(next))); if (candidate < cost.getOrDefault(next, Double.POSITIVE_INFINITY)) { cost.put(next, candidate); prev.put(next, current.node); queue.add(new Entry(next, candidate)); } } }
        if (!cost.containsKey(end)) return Collections.emptyList(); List<double[]> output = new ArrayList<>(); for (String node=end; node != null; node=prev.get(node)) output.add(xy(geoNodes.get(node))); Collections.reverse(output); return output;
    }
    /**
     * Match against the bundled road graph without network access. This is a
     * conservative local candidate scorer, not an unconditional visual snap:
     * confidence falls on distant roads, heading disagreement, or close rivals.
     */
    Match match(double east, double north, double heading, double positionSigmaM) {
        if (!isReady() || !Double.isFinite(east) || !Double.isFinite(north) || !Double.isFinite(heading)) return null;
        double bestScore=0,bestE=0,bestN=0,bestHeading=0,bestDistance=Double.POSITIVE_INFINITY,secondScore=0;
        double sigma=Math.max(4,Math.min(80,positionSigmaM));
        for(double[][] way:roads()) for(int i=0;i<way.length-1;i++) {
            double[] a=way[i],b=way[i+1]; double de=b[0]-a[0],dn=b[1]-a[1],length=de*de+dn*dn;
            if(length<.01) continue;
            double ratio=Math.max(0,Math.min(1,((east-a[0])*de+(north-a[1])*dn)/length));
            double pe=a[0]+ratio*de,pn=a[1]+ratio*dn,distance=Math.hypot(east-pe,north-pn);
            double roadHeading=Math.atan2(de,dn),headingError=Math.abs(wrap(heading-roadHeading));
            // Directional driving avoids accepting an opposite-direction road.
            if(headingError>Math.PI/2) { roadHeading=wrap(roadHeading+Math.PI); headingError=Math.abs(wrap(heading-roadHeading)); }
            double score=Math.exp(-.5*(distance/sigma)*(distance/sigma))*Math.exp(-.5*(headingError/.55)*(headingError/.55));
            if(score>bestScore){secondScore=bestScore;bestScore=score;bestE=pe;bestN=pn;bestHeading=roadHeading;bestDistance=distance;} else if(score>secondScore) secondScore=score;
        }
        if(bestScore==0 || bestDistance>Math.max(25,2.5*sigma)) return null;
        double ambiguity=Math.max(0,1-secondScore/Math.max(bestScore,1e-9));
        return new Match(bestE,bestN,bestHeading,bestScore*ambiguity,bestDistance);
    }
    private void connect(String a, double[] aPoint, String b, double[] bPoint) { geoNodes.put(a, aPoint); geoNodes.put(b, bPoint); edges.computeIfAbsent(a, ignored -> new ArrayList<>()).add(b); edges.computeIfAbsent(b, ignored -> new ArrayList<>()).add(a); }
    private String closest(double east, double north) { String best = null; double bestDistance = Double.POSITIVE_INFINITY; for (Map.Entry<String,double[]> entry : geoNodes.entrySet()) { double d = distance(xy(entry.getValue()), new double[]{east,north}); if (d < bestDistance) { bestDistance=d; best=entry.getKey(); } } return best; }
    private double[] xy(double[] geo) { return new double[]{R*Math.toRadians(geo[1]-lon0)*Math.cos(Math.toRadians(lat0)), R*Math.toRadians(geo[0]-lat0)}; }
    private static String key(double[] point) { return String.format(java.util.Locale.US, "%.7f,%.7f", point[0], point[1]); }
    private static double distance(double[] a, double[] b) { return Math.hypot(a[0]-b[0],a[1]-b[1]); }
    private static double wrap(double radians) { return Math.atan2(Math.sin(radians),Math.cos(radians)); }
    private static final class Entry implements Comparable<Entry> { final String node; final double cost; Entry(String n,double c){node=n;cost=c;} public int compareTo(Entry other){return Double.compare(cost,other.cost);} }
}
