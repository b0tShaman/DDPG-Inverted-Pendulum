package src.main.java.com.ml.algorithms.DDPG;

import javax.swing.*;
import java.awt.*;

public class Pendulum2 extends JPanel implements Runnable {

    private double theta = Math.PI / 2;

    private double x = Math.PI / 2;
    private double y = Math.PI / 2;

    private int length;

    public Pendulum2(int length) {
        this.length = length * 100;
        setDoubleBuffered(true);
    }

    public void setState( Environment_Pendulum.State state){
        this.theta = state.theta;
        this.x = state.x;
        this.y = state.y;
    }

    @Override
    public void paint(Graphics g) {
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, getWidth(), getHeight());
        g.setColor(Color.BLACK);
        int anchorX = getWidth()/2; // Math.round((float)x * 100) + (getWidth() / 2);
        int anchorY = getHeight()/2;// -((getHeight()/2) - 10) ;
        int ballX = anchorX + (int) (Math.sin(theta) * length);
        int ballY = anchorY - (int) (Math.cos(theta) * length) ;
        g.drawLine(0, getHeight()/2, getWidth(), getHeight()/2);
        g.drawLine(anchorX, anchorY, ballX, ballY);
        g.fillOval(anchorX-10 , anchorY- 10 , 20, 20);
        g.fillOval(ballX - 7, ballY - 7, 14, 14);
    }

    public void run() {
        while(true) {
            repaint();
        }
    }

    @Override
    public Dimension getPreferredSize() {
        return new Dimension(2 * 400 + 50, 400 / 2 * 3);
    }

}
