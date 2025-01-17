package src.main.java.com.ml.algorithms.DDPG;

public class Environment_Pendulum {
    public int actionSpace = DDPG.maxTorque;
    public double theta_current;
    public double theta_dot_current;
    public double Length_Of_Pendulum = DDPG.Length_Of_Pendulum;//0.326;
    public double Mass_Of_Pendulum = DDPG.Mass_Of_Pendulum;
    public double g = DDPG.g; // acceleration due to gravity
    public double t = 0;
    public double dt = 0.05;

    public static class State {
        public double x;
        public double y;
        public double x_dot;
        public double theta;
        public double theta_dot;

        public double time;
        public double theta_degree;
        public double reward;

        public State( double theta, double theta_dot, double time, double reward) {
            this.x = Math.cos(theta);
            this.y = Math.sin(theta);
            this.x_dot = x_dot;
            this.theta = theta;
            this.theta_dot = theta_dot;
            this.time = time;
            theta_degree = (this.theta * (180 / Math.PI));
            this.reward = reward;
        }

        @Override
        public String toString() {
            return "x = " + x + ", y = " + y + ", theta_dot = " + theta_dot + ", reward = " + reward + ", time = " + time + ", Theta degree = " + theta_degree;
        }
    }

    public Environment_Pendulum(double O_Initial, double theta_dot_initial){
        //System.out.println("O_Initial == " + O_Initial + " theta_dot_initial " + theta_dot_initial);
        this.theta_current = O_Initial;
        this.theta_dot_current = theta_dot_initial;
    }

    public double clip( double tau, double maxMag){
        if ( tau > maxMag)return maxMag;
        else return Math.max(tau, (-1 * maxMag));
    }

    public double normalize( double x){
        if( x > Math.PI)
            x -= 2.0 * Math.PI;

        if ( x < -Math.PI)
            x += 2.0 * Math.PI;

       return x;
    }

    public State getNewStateAndReward(double torque){
        //System.out.println("force " + force);
        //double angular_acc = (Math.sin(theta) * (g/L)) - (acceleration * (Math.cos(theta)/L)) ;

        // Open AI pendulum
        double u = clip(torque,actionSpace);
        double costs = Math.pow((theta_current) , 2) + (0.1 * Math.pow(theta_dot_current,2)) + (0.001 * (Math.pow(u,2)));

        //double angular_acc = ( ( (M+m)*g*Math.sin(theta)) - ( (Math.cos(theta) * ( force + (m * L * Math.pow(omega,2)*Math.sin(theta)) ) ) ) )/( ( (M+m) * L) - ( m * L * Math.pow(Math.cos(theta),2)));
        //double cart_acceleration = (((m * angular_acc) - (g*Math.sin(theta)))/(Math.cos(theta)));

        //double angular_acc = ( ( (M+m)*g*Math.sin(theta)) - ( (Math.cos(theta) * ( force + (m * L * Math.pow(omega,2)*Math.sin(theta)) ) ) ) )/( ((4/3.0) * (M+m) * L) - ( m * L * Math.pow(Math.cos(theta),2)));
        //double cart_acceleration = ( force + (m*L* (  (Math.pow(omega,2)*Math.sin(theta) ) - (angular_acc * Math.cos(theta))) ))/(M+m);

        theta_dot_current = theta_dot_current + ( ( ((3 * g) / (2 * Length_Of_Pendulum)) * Math.sin(theta_current)) + (3.0 / (Mass_Of_Pendulum *  Math.pow(Length_Of_Pendulum,2))) * u) * dt;
        theta_dot_current = clip(theta_dot_current,8);
        theta_current = theta_current + theta_dot_current * dt;
        theta_current = normalize(theta_current);

        t= t + dt;

        return new State(theta_current, theta_dot_current, t, (-1 * costs));
    }

    public State getState(){
        double costs = Math.pow((theta_current) , 2) + 0.1 * Math.pow(theta_dot_current,2) + 0.001 * (Math.pow(0,2));
        return new State(theta_current, theta_dot_current, t, (-1 * costs));
    }
}
