package in.navrakshak.idr;

import android.content.Context;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.BufferedReader;
import java.io.InputStreamReader;

/** Dependency-free local inference for the 1,273-parameter IO-VNBD SpeedTCN. */
final class SpeedTcnInference {
    static final String OUTPUT_SEMANTICS = SpeedTcnContract.OUTPUT_SEMANTICS;
    private static final String[] FEATURES = SpeedTcnContract.FEATURES;
    private final double[] mean=new double[9], std=new double[9], b1=new double[24], bd=new double[24], linear=new double[24];
    private final double[][][] w1=new double[24][9][5]; private final double[][] wd=new double[24][5]; private double linearBias;
    private final double[][] history=new double[20][9]; private int count;
    SpeedTcnInference(Context context) {
        try (BufferedReader reader=new BufferedReader(new InputStreamReader(context.getAssets().open("speed_tcn_all_io_vnbd.json")))) {
            StringBuilder text=new StringBuilder();String line;while((line=reader.readLine())!=null)text.append(line);JSONObject root=new JSONObject(text.toString()), weights=root.getJSONObject("weights");
            if (!"navrakshak-speed-tcn-v1".equals(root.optString("format")) || root.optInt("window",-1)!=SpeedTcnContract.WINDOW_SAMPLES) throw new IllegalArgumentException("Unsupported speed model format/window");
            // Older generated assets predate this metadata. They are accepted
            // only as this explicitly known delta-speed format; all new exports
            // carry the field and any other semantic is rejected.
            if (!OUTPUT_SEMANTICS.equals(root.optString("output_semantics",OUTPUT_SEMANTICS))) throw new IllegalArgumentException("Speed model is not a delta-speed model");
            JSONArray declaredFeatures=root.getJSONArray("features");
            if(declaredFeatures.length()!=FEATURES.length) throw new IllegalArgumentException("Unexpected speed feature count");
            for(int i=0;i<FEATURES.length;i++) if(!FEATURES[i].equals(declaredFeatures.getString(i))) throw new IllegalArgumentException("Unexpected speed feature order at "+i);
            copy(root.getJSONArray("mean"),mean);copy(root.getJSONArray("std"),std);copy(weights.getJSONArray("network.0.bias"),b1);copy(weights.getJSONArray("network.2.bias"),bd);copy(weights.getJSONArray("network.6.weight").getJSONArray(0),linear);linearBias=weights.getJSONArray("network.6.bias").getDouble(0);
            JSONArray first=weights.getJSONArray("network.0.weight"), depth=weights.getJSONArray("network.2.weight");
            for(int o=0;o<24;o++){for(int f=0;f<9;f++)copy(first.getJSONArray(o).getJSONArray(f),w1[o][f]);copy(depth.getJSONArray(o).getJSONArray(0),wd[o]);}
        } catch(Exception error) { throw new IllegalStateException("Unable to load local AI model",error); }
    }
    double update(float[] features) {
        for(int i=0;i<19;i++)System.arraycopy(history[i+1],0,history[i],0,9);
        for(int f=0;f<9;f++)history[19][f]=(features[f]-mean[f])/Math.max(1e-4,std[f]); if(count<20)count++; if(count<20)return Double.NaN;
        double[][] channels=new double[24][20];
        for(int o=0;o<24;o++)for(int t=0;t<20;t++){double sum=b1[o];for(int f=0;f<9;f++)for(int k=0;k<5;k++){int source=t+k-2;if(source>=0&&source<20)sum+=w1[o][f][k]*history[source][f];}channels[o][t]=Math.max(0,sum);}
        double output=linearBias;
        for(int o=0;o<24;o++){double average=0;for(int t=0;t<20;t++){double sum=bd[o];for(int k=0;k<5;k++){int source=t+k-2;if(source>=0&&source<20)sum+=wd[o][k]*channels[o][source];}average+=Math.max(0,sum);}output+=linear[o]*(average/20);}
        return output;
    }
    void reset() {
        for (double[] sample : history) java.util.Arrays.fill(sample, 0);
        count=0;
    }
    private static void copy(JSONArray input,double[] output)throws Exception{for(int i=0;i<output.length;i++)output[i]=input.getDouble(i);}
}
