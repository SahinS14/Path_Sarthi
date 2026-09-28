package in.navrakshak.idr;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.view.View;

final class PipelineView extends View {
    private final Paint line=new Paint(Paint.ANTI_ALIAS_FLAG), text=new Paint(Paint.ANTI_ALIAS_FLAG);
    PipelineView(Context c){super(c);line.setColor(Color.rgb(30,88,137));line.setStrokeWidth(3);text.setColor(Color.rgb(148,190,230));text.setTextSize(13);text.setTextAlign(Paint.Align.CENTER);}
    @Override protected void onDraw(Canvas c){int w=getWidth();float y=25;float[] xs={w*.08f,w*.29f,w*.50f,w*.72f,w*.92f};for(int i=0;i<xs.length-1;i++)c.drawLine(xs[i],y,xs[i+1],y,line);for(float x:xs){c.drawCircle(x,y,5,line);c.drawCircle(x,y,2,new Paint(line));}String[] labels={"AI","INS","MAP","YOU"};for(int i=0;i<labels.length;i++)c.drawText(labels[i],xs[i+1],55,text);}
}
