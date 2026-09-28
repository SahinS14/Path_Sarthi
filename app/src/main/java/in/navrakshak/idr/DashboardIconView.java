package in.navrakshak.idr;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;
import android.view.View;

/** Lightweight, original line icons for the Path Sārathi dashboard cards. */
final class DashboardIconView extends View {
    private static final int ACCENT=Color.rgb(235,195,110),CUTOUT=Color.rgb(19,35,55); private final int type; private final Paint paint=new Paint(Paint.ANTI_ALIAS_FLAG);
    DashboardIconView(Context c,int type){super(c);this.type=type;paint.setStrokeWidth(6);paint.setStrokeCap(Paint.Cap.ROUND);paint.setStyle(Paint.Style.STROKE);paint.setColor(ACCENT);}
    @Override protected void onDraw(Canvas c){super.onDraw(c);float w=getWidth(),h=getHeight(),x=w/2,y=h/2; if(type==0){paint.setStyle(Paint.Style.FILL);c.drawCircle(x,y-13,11,paint);Path p=new Path();p.moveTo(x,y+27);p.lineTo(x-22,y-4);p.lineTo(x+22,y-4);p.close();c.drawPath(p,paint);paint.setColor(CUTOUT);c.drawCircle(x,y-13,4,paint);paint.setColor(ACCENT);paint.setStyle(Paint.Style.STROKE);}else if(type==1){c.drawRoundRect(new RectF(x-20,y-28,x+20,y+28),5,5,paint);c.drawLine(x+7,y-28,x+20,y-15,paint);c.drawLine(x-11,y-7,x+11,y-7,paint);c.drawLine(x-11,y+6,x+11,y+6,paint);c.drawLine(x-11,y+19,x+3,y+19,paint);}else if(type==2){c.drawArc(new RectF(x-24,y-28,x+24,y+36),180,-180,false,paint);c.drawArc(new RectF(x-15,y-18,x+15,y+36),180,-180,false,paint);c.drawLine(x-24,y+36,x+24,y+36,paint);}else{paint.setStyle(Paint.Style.FILL);for(int i=0;i<8;i++){double a=i*Math.PI/4;float px=(float)(x+23*Math.cos(a)),py=(float)(y+23*Math.sin(a));c.drawRoundRect(new RectF(px-5,py-10,px+5,py+10),3,3,paint);}c.drawCircle(x,y,24,paint);paint.setColor(CUTOUT);c.drawCircle(x,y,9,paint);paint.setColor(ACCENT);paint.setStyle(Paint.Style.STROKE);}}
}
