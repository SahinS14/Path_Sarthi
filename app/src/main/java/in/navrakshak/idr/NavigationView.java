package in.navrakshak.idr;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.view.MotionEvent;
import android.view.View;
import java.util.ArrayList;
import java.util.List;

/** Offline OSM map, route, live IDR trail, and tap-to-select destination. */
public final class NavigationView extends View {
    private final List<double[]> trail = new ArrayList<>(); private final OfflineRoadMap roads;
    private final Paint road = paint(Color.rgb(71,85,105), 3), routePaint = paint(Color.rgb(168,85,247), 7), trailPaint = paint(Color.rgb(56,189,248), 5), vehicle = paint(Color.GREEN, 1), marker = paint(Color.rgb(251,191,36), 1), label = paint(Color.LTGRAY, 1);
    private double east, north, heading, sourceE, sourceN, destinationE=Double.NaN, destinationN=Double.NaN, centerE, centerN, scale=1; private String mode="WAITING_FOR_GNSS"; private List<double[]> route = new ArrayList<>();
    public NavigationView(Context context) { super(context); roads=new OfflineRoadMap(context); label.setTextSize(26); setClickable(true); }
    private static Paint paint(int color, float width) { Paint p=new Paint(Paint.ANTI_ALIAS_FLAG); p.setColor(color); p.setStrokeWidth(width); p.setStrokeCap(Paint.Cap.ROUND); p.setStyle(Paint.Style.STROKE); return p; }
    public void setMapOrigin(double latitude, double longitude) { roads.setOrigin(latitude, longitude); invalidate(); }
    public void update(double newEast, double newNorth, double newHeading, String newMode) { east=newEast; north=newNorth; heading=newHeading; mode=newMode; if(trail.isEmpty()){sourceE=east;sourceN=north;} if(trail.isEmpty()||Math.hypot(east-trail.get(trail.size()-1)[0],north-trail.get(trail.size()-1)[1])>.25)trail.add(new double[]{east,north}); if(trail.size()>1200)trail.remove(0); invalidate(); }
    @Override public boolean onTouchEvent(MotionEvent event) { if (event.getAction()==MotionEvent.ACTION_UP && !trail.isEmpty()) { destinationE=centerE+(event.getX()-getWidth()/2.0)/scale; destinationN=centerN-(event.getY()-getHeight()/2.0)/scale; route=roads.route(east,north,destinationE,destinationN); invalidate(); return true; } return true; }
    @Override protected void onDraw(Canvas canvas) {
        super.onDraw(canvas); canvas.drawColor(Color.rgb(15,30,48)); int w=getWidth(),h=getHeight();
        double minE=east-150,maxE=east+150,minN=north-150,maxN=north+150;
        for(double[][] way:roads.roads()) for(double[] point:way){minE=Math.min(minE,point[0]);maxE=Math.max(maxE,point[0]);minN=Math.min(minN,point[1]);maxN=Math.max(maxN,point[1]);}
        if(!Double.isNaN(destinationE)){minE=Math.min(minE,destinationE);maxE=Math.max(maxE,destinationE);minN=Math.min(minN,destinationN);maxN=Math.max(maxN,destinationN);}
        double span=Math.max(220,Math.max(maxE-minE,maxN-minN)*1.12); centerE=(minE+maxE)/2; centerN=(minN+maxN)/2; scale=Math.min(w,h)/span;
        for(double[][] way:roads.roads()) drawLine(canvas,way,road,w,h); if(!route.isEmpty())drawLine(canvas,route.toArray(new double[0][]),routePaint,w,h); if(trail.size()>1)drawLine(canvas,trail.toArray(new double[0][]),trailPaint,w,h);
        canvas.drawText(roads.isReady()?"OFFLINE OSM • TAP MAP TO SET DESTINATION":"WAITING FOR DEMO MAP AREA",18,34,label); canvas.drawText(route.isEmpty()?"Tap a road to choose destination":"Route ready • "+mode,18,h-18,label);
        if(!Double.isNaN(destinationE)){marker.setStyle(Paint.Style.FILL);canvas.drawCircle(x(destinationE,w),y(destinationN,h),11,marker);marker.setStyle(Paint.Style.STROKE);}
        if(trail.isEmpty())return; float vx=x(east,w),vy=y(north,h); canvas.save();canvas.rotate((float)Math.toDegrees(heading),vx,vy); Path arrow=new Path();arrow.moveTo(vx,vy-22);arrow.lineTo(vx-14,vy+16);arrow.lineTo(vx+14,vy+16);arrow.close();vehicle.setStyle(Paint.Style.FILL);vehicle.setColor(mode.equals("GNSS_AIDED")?Color.rgb(34,197,94):Color.rgb(251,191,36));canvas.drawPath(arrow,vehicle);canvas.restore();
    }
    private void drawLine(Canvas c,double[][] points,Paint p,int w,int h){if(points.length<2)return;Path path=new Path();for(int i=0;i<points.length;i++){float px=x(points[i][0],w),py=y(points[i][1],h);if(i==0)path.moveTo(px,py);else path.lineTo(px,py);}c.drawPath(path,p);}
    private float x(double e,int w){return(float)(w/2+(e-centerE)*scale);} private float y(double n,int h){return(float)(h/2-(n-centerN)*scale);}
}
