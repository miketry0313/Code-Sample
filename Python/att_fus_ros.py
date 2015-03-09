#!/usr/bin/env python
"""
Created on Feb 9 2015   
@author: Ruoyu Tan
Credits is given to Nathan Depenbusch

This code is developed for building an orientation estimator for AUV project by
Uncented Kalman Filter.
"""
import rospy
import numpy 
import scipy.io
import math

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Vector3Stamped

class UnscentedKF():
    """Uncented KalmanFilter 

    Attributes:
        sys_dynamics,sys_dyn: system dynamics function, a attribute from
        mea_dynamics,mea_dyn: measurement dynamics function, a attribute from
        P_INITIAL, P_INIT: Initial value of error covariance
        X_INIT: Initial value of system states
        p_cov: error covirance matrix (n by n)
        p_prior: prior error covariance (n by n)
        x_hat: state estiamte
        x_chi_cur: state sigma points by giving k-1
        CONST_N: number of sigma points
        CONST_LAMBDA: scaling parameter
        W_C,W_M: corresponding weight
        
    """
    def __init__(self,sys_dyn,mea_dyn,P_INIT,X_INIT):
        """Initiates Uncented Kalman Filter """
        self.p_cov=P_INIT        
        self.p_prior=P_INIT
        self.x_hat=X_INIT
        self.x_hat_prior=X_INIT
        self.P_INITIAL=P_INIT      
        self.sys_dynamics=sys_dyn
        self.mea_dynamics=mea_dyn
        self.x_chi_cur=numpy.zeros([4,2*X_INIT.size+1],dtype=float)
        CONST_ALPHA=1   #Give a value for the spread of sigma point
        CONST_BETA=2   #Give a value for the prior knowledge of distribution
        self.CONST_N=self.x_hat.size   
        self.CONST_LAMBDA=CONST_ALPHA*numpy.sqrt(self.CONST_N)
        self.W_M=numpy.zeros([2*self.CONST_N+1,1],dtype=float)
        self.W_M[0][0]=((CONST_ALPHA**2)*(self.CONST_N)-self.CONST_N)/(self.CONST_N+self.CONST_LAMBDA)
        for i in range(0,2*self.CONST_N+1):
            self.W_M[i]=1/(2*self.CONST_N+2*self.CONST_LAMBDA)       
        self.W_C=numpy.eye(2*self.CONST_N+1,dtype=float)
        self.W_C[0][0]=self.CONST_LAMBDA/(self.CONST_N+self.CONST_LAMBDA)+(1-CONST_ALPHA**2+CONST_BETA)
        for i in range(0,2*self.CONST_N+1):
            self.W_C[i][i]=self.W_M[i]
            
    def time_update(self,system_input,Q_COV):
        """Time Update Part in UKF

        Args:
            system_input: system input
            Q_COV: system noise covariance matrix
        """
        # find cholesky but check for non positive definite matrices
        # need to calculate eignvalue!!! show eigenvalue greater than zero
        x_chi_sigma=numpy.zeros([4,2*self.CONST_N+1],dtype=float)
        eigenvalue_p,eigenvector_p=numpy.linalg.eig(self.p_cov)
        for i in range(0,self.CONST_N):
            if eigenvalue_p[i]<0:
                self.p_cov=self.P_INITIAL
                break
        l_chol= numpy.linalg.cholesky(self.p_cov) 
        # Calculate sigma points
        for i in range(2*self.CONST_N+1):
            if i==0:
                x_chi_sigma[:,i]=self.x_hat.flatten()   
            elif i<self.CONST_N+1:
                x_chi_sigma[:,i]=self.x_hat.flatten()+l_chol[:,i-1]*self.CONST_LAMBDA
            else:
                x_chi_sigma[:,i]=self.x_hat.flatten()-l_chol[:,i-self.CONST_N-1]*self.CONST_LAMBDA
        #Calculate state sigma points by giving k-1
        self.x_chi_cur=self.sys_dynamics(system_input,x_chi_sigma)
        #Obtain a priori estimate
        self.x_hat_prior=numpy.dot(self.x_chi_cur,self.W_M)
        #Estimate a prior error covariance
        temp_prior = self.x_chi_cur-numpy.tile(self.x_hat_prior, (1, (2*self.CONST_N+1)))
        self.p_prior=numpy.dot(temp_prior,numpy.dot(self.W_C,temp_prior.T))+Q_COV
        
    def measurement_update(self,mea_input,R_COV,mea_ref):
        """Measurement Update Part in UKF

        Args:
            mea_input: measurement input
            R_COV: measurement noise covariance matrix
            mea_ref: measurement reference points
        """
        #Calculate measurement sigma points by giving k-1
        y_chi_prior=self.mea_dynamics(self.x_chi_cur,self.CONST_N,mea_ref)
        #Obtain a predicted measurement
        y_hat_prior=numpy.dot(y_chi_prior,self.W_M)
        temp_x = self.x_chi_cur-numpy.tile(self.x_hat_prior, (1, (2*self.CONST_N+1)))
        temp_y = y_chi_prior-numpy.tile(y_hat_prior, (1, (2*self.CONST_N+1)))
        #Estimate the covariance of y        
        p_yy=numpy.dot(temp_y,numpy.dot(self.W_C,temp_y.T))+R_COV
        #Estimate the covariance between x and y        
        p_xy=numpy.dot(temp_x,numpy.dot(self.W_C,temp_y.T))
        #Calculate Kalman Filter gain
        k_kalman = numpy.dot(p_xy, numpy.linalg.inv(p_yy))
        #The measurement update of the state estimate
        self.x_hat=self.x_hat+numpy.dot(k_kalman,(mea_input-y_hat_prior))
        #Estimate a posterior error covariance
        self.p_cov=self.p_prior-numpy.dot(k_kalman,numpy.dot(p_yy,k_kalman.T))
     
class AttitudeFilter():
    """Attitude estimator for AUV by fusing IMU data 

    Attributes:
        P_INIT: Initial value of error covariance
        X_INIT: Initial value of system states(4 by 1 quatonions)
        Q_EUL: System noise covariance matrix (4 by 4)
        uncented_kf: instance of uncented kalman filter class
        x_chi_cur: state sigma points by giving k-1
        dt: time interval between update system input
    """

    def __init__(self,att_pub):
        """Initiates AttitudeFilter Classes"""
        P_INIT=1000.0*numpy.eye(4,dtype=float)     
        self.X_INIT=numpy.zeros([4,1],dtype=float)   
        self.X_INIT[0][0]=1
        self.Q_EUL=numpy.eye(4,dtype=float)   
        self.Q_EUL[0][0]=0.01   # Uncertainty in quatnion when system subject flucturation
        self.Q_EUL[1][1]=0.01   
        self.Q_EUL[2][2]=0.01   
        self.Q_EUL[3][3]=0.01   
        # initialize a UKF with this class's members
        self.uncented_kf= UnscentedKF(self.system_dynamics, self.measurement_dynamics,P_INIT,self.X_INIT)
        self.x_chi_cur=numpy.zeros([4,2*self.X_INIT.size+1],dtype=float)
        self.last_update=0
        self.dt=0.02
        self.att_pub=att_pub
        self.timer1=0
        self.timer2=0
        self.timer_limit=5000
        self.out_confi=numpy.zeros((self.timer_limit,4))
        self.out_state=numpy.zeros((self.timer_limit,4))
        self.out_int=numpy.zeros((self.timer_limit,4))
        self.out_euler1=numpy.zeros((self.timer_limit,3))
        self.out_euler2=numpy.zeros((self.timer_limit,3))
        self.out_acc=numpy.zeros((self.timer_limit,3))
     
    def system_dynamics(self,gyro_input,x_chi_sigma):
        """System dynamics function: 
        Calculating state sigma points by giving k-1 through system matrix

        Args:
            gyro_input: Angular rate volocity from raw measurement from Gyro (3 by 1,unit:rad/s)
            x_chi_sigma: Sigma point of state (4 by 2*n+1) 
        
        Returns:
            x_chi_cur: sigma points by giving k-1
        """
        #now=rospy.get_time()
        #dt=now-self.last_update  
        #Define system matrix
        sys_matrix=numpy.zeros([4,4],dtype=float)
        sys_matrix[0][0]=1
        sys_matrix[0][1]=gyro_input[0]*self.dt/2
        sys_matrix[0][2]=gyro_input[1]*self.dt/2
        sys_matrix[0][3]=gyro_input[2]*self.dt/2
        sys_matrix[1][0]=-gyro_input[0]*self.dt/2
        sys_matrix[1][1]=1
        sys_matrix[1][2]=-gyro_input[2]*self.dt/2
        sys_matrix[1][3]=gyro_input[1]*self.dt/2
        sys_matrix[2][0]=-gyro_input[1]*self.dt/2
        sys_matrix[2][1]=gyro_input[2]*self.dt/2
        sys_matrix[2][2]=1
        sys_matrix[2][3]=-gyro_input[0]*self.dt/2
        sys_matrix[3][0]=-gyro_input[2]*self.dt/2
        sys_matrix[3][1]=-gyro_input[1]*self.dt/2
        sys_matrix[3][2]=gyro_input[0]*self.dt/2
        sys_matrix[3][3]=1
        
        self.x_chi_cur=numpy.dot(sys_matrix,x_chi_sigma)  
        #self.last_update=now
        return self.x_chi_cur
    
    def measurement_dynamics(self,chi_cur,CONST_N,INERTIAL_COM):
        """Measument dynamics function: 
        Calculating measurement sigma points by giving k-1 through system matrix

        Args:
            chi_cur: Sigma point of state (4 by 2*n+1) 
            CONST_N: scaling parameter
            INERTIAL_COM: accleration(m/sec^2) or mag value in inertial frame(T) (3 by 1)
        
        Returns:
            y_chi_cur: measurement sigma points by giving k-1
        """
        y_chi_cur=numpy.zeros([3,2*self.X_INIT.size+1],dtype=float)
        for i in range(2*CONST_N+1):
            mea_matrix=numpy.zeros([3,3],dtype=float)
            mea_matrix[0][0]=chi_cur[0][i]**2+chi_cur[1][i]**2 \
                            -chi_cur[2][i]**2-chi_cur[3][i]**2
            mea_matrix[0][1]=2*(numpy.dot(chi_cur[1][i],chi_cur[2][i]) \
                             +numpy.dot(chi_cur[0][i],chi_cur[3][i]))            
            mea_matrix[0][2]=2*(numpy.dot(chi_cur[1][i],chi_cur[3][i]) \
                             +numpy.dot(chi_cur[0][i],chi_cur[2][i]))                       
            mea_matrix[1][0]=2*(numpy.dot(chi_cur[1][i],chi_cur[2][i]) \
                             +numpy.dot(chi_cur[0][i],chi_cur[3][i]))
            mea_matrix[1][1]=chi_cur[0][i]**2-chi_cur[1][i]**2 \
                            +chi_cur[2][i]**2-chi_cur[3][i]**2
            mea_matrix[1][2]=2*(numpy.dot(chi_cur[2][i],chi_cur[3][i]) \
                             -numpy.dot(chi_cur[0][i],chi_cur[1][i]))
            mea_matrix[2][0]=2*(numpy.dot(chi_cur[1][i],chi_cur[3][i]) \
                             +numpy.dot(chi_cur[0][i],chi_cur[2][i]))            
            mea_matrix[2][1]=2*(numpy.dot(chi_cur[2][i],chi_cur[3][i]) \
                             +numpy.dot(chi_cur[0][i],chi_cur[1][i]))                       
            mea_matrix[2][2]=chi_cur[0][i]**2-chi_cur[1][i]**2 \
                             -chi_cur[2][i]**2+chi_cur[3][i]*2
            temp=numpy.dot(mea_matrix,INERTIAL_COM)
            y_chi_cur[:,i]=temp.flatten()
        return y_chi_cur
                
    def gyro_update(self,data):
        """: 
        When new system input comes, trigger Time Update process

        Args:
            data: Angular rate volocity from raw measurement from Gyro (3 by 1,unit:rad/s)       
        """
        #gyro_mea=[data.vector.x,data.vector.y,data.vector.z]
        gyro_mea=numpy.zeros([3,1],dtype=float)
        gyro_mea[0][0]=data.vector.x
        gyro_mea[1][0]=data.vector.y
        gyro_mea[2][0]=data.vector.z
        gyro_qua=self.euler2_qua(gyro_mea)
        #Trigger time update
        (qua_prior,p_prior)=self.uncented_kf.time_update(gyro_qua,self.Q_EUL)
        flag=math.pow(qua_prior[0],2)+math.pow(qua_prior[1],2)+math.pow(qua_prior[2],2)+math.pow(qua_prior[3],2)
        #Normalize estimated result 
        if flag>1:
            norm=math.sqrt(qua_prior[0]**2+qua_prior[1]**2+qua_prior[2]**2+qua_prior[3]**2)
            qua_prior[0]=qua_prior[0]/norm
            qua_prior[1]=qua_prior[1]/norm
            qua_prior[2]=qua_prior[2]/norm
            qua_prior[3]=qua_prior[3]/norm      
        
    def acc_update(self,data):
        """
        When new measurement input comes, trigger Measurement Update process

        Args:
            data: Acceleration from raw measurement from acc (3 by 1,unit:meter/sec^2)       
        """
        #acc_mea=[data.vector.x,data.vector.y,data.vector.z]
        acc_mea=numpy.zeros([3,1],dtype=float)
        acc_mea[0][0]=data.vector.x
        acc_mea[1][0]=data.vector.y
        acc_mea[2][0]=data.vector.z
        INERTIAL_COM=numpy.zeros([3,1],dtype=float)
        R_EUL=numpy.zeros([3,3],dtype=float)
        INERTIAL_COM[0][0]=0
        INERTIAL_COM[1][0]=0
        INERTIAL_COM[2][0]=0.98
        #start at all equal to 0.01
        R_EUL[0][0]=0.05   # Covariance error for acclometer in x direction
        R_EUL[1][1]=0.05   # Covariance error for acclometer in y direction
        R_EUL[2][2]=0.05
        #Trigger measurement update update
        (est_qua,est_p)=self.uncented_kf.measurement_update(acc_mea,R_EUL,INERTIAL_COM)
        #Normalize estimated result 
        flag=math.pow(est_qua[0],2)+math.pow(est_qua[1],2)+math.pow(est_qua[2],2)+math.pow(est_qua[3],2)
        if flag>1:
            norm=math.sqrt(est_qua[0]**2+est_qua[1]**2+est_qua[2]**2+est_qua[3]**2)
            est_qua[0]=est_qua[0]/norm
            est_qua[1]=est_qua[1]/norm
            est_qua[2]=est_qua[2]/norm
            est_qua[3]=est_qua[3]/norm
        
    def mag_update(self,data):
        """
        When new magnometer measurement input comes, trigger Measurement Update process

        Args:
            data: Magnometer from raw measurement from mag (3 by 1,unit:?)       
        """
        mag_mea=[data.vector.x,data.vector.y,data.vector.z]
        INERTIAL_COM=numpy.zeros([3,1],dtype=float)     
        R_EUL=numpy.zeros([3,3],dtype=float)   
        INERTIAL_COM[0][0]=0.00001976
        INERTIAL_COM[0][1]=-0.000003753
        INERTIAL_COM[0][2]=0.00004858
        R_EUL[0][0]=0.01   # Covariance error for magnometerin x direction
        R_EUL[1][1]=0.01   # Covariance error for magnometer in y direction
        R_EUL[2][2]=0.01
        #Trigger measurement update
        est_qua=self.uncented_kf.measurement_update(mag_mea,R_EUL,INERTIAL_COM)
        flag=math.pow(est_qua[0],2)+math.pow(est_qua[1],2)+math.pow(est_qua[2],2)+math.pow(est_qua[3],2)
        if flag>1:
            norm=math.sqrt(est_qua[0]**2+est_qua[1]**2+est_qua[2]**2+est_qua[3]**2)
            est_qua[0]=est_qua[0]/norm
            est_qua[1]=est_qua[1]/norm
            est_qua[2]=est_qua[2]/norm
            est_qua[3]=est_qua[3]/norm
    
    def euler2_qua(self,euler_angle):
        """
        Convert Euler angle to quatenion angle

        Args:
            euler_angle: pitch, roll, yaw angles in Euler formats   
        
        Returns:
            qua_angle: return quatenion format
        """
        qua_angle=numpy.zeros([4,1],dtype=float)     
        qua_angle[0]=(math.cos(euler_angle[0]/2)*math.cos(euler_angle[1]/2)*math.cos(euler_angle[2]/2)) \
                  +(math.sin(euler_angle[0]/2)*math.cos(euler_angle[1]/2)*math.cos(euler_angle[2]/2))
        qua_angle[1]=(math.sin(euler_angle[0]/2)*math.cos(euler_angle[1]/2)*math.cos(euler_angle[2]/2)) \
                  -(math.cos(euler_angle[0]/2)*math.sin(euler_angle[1]/2)*math.sin(euler_angle[2]/2))
        qua_angle[2]=(math.cos(euler_angle[0]/2)*math.sin(euler_angle[1]/2)*math.cos(euler_angle[2]/2)) \
                  +(math.sin(euler_angle[0]/2)*math.cos(euler_angle[1]/2)*math.sin(euler_angle[2]/2))
        qua_angle[3]=(math.cos(euler_angle[0]/2)*math.cos(euler_angle[1]/2)*math.sin(euler_angle[2]/2)) \
                  -(math.sin(euler_angle[0]/2)*math.sin(euler_angle[1]/2)*math.cos(euler_angle[2]/2))
        return qua_angle
    
    def qua2_euler(self,qua_angle):
        """
        Convert quatenion angle to euler angle

        Args:
            qua_angle: pitch, roll, yaw angles in quatenion formats   
        
        Returns:
            qua_angle: return values in Euler format
        """    
        euler_angle=numpy.zeros([3,1],dtype=float)    
        #TODO: need to check atan2(x,y) or atan2(y,x)
        euler_angle[0]=math.atan2(2*(qua_angle[0]*qua_angle[1]+qua_angle[2]*qua_angle[3]),1-2*(qua_angle[1]**2+qua_angle[2]**2))
        euler_angle[1]=math.asin(2*(qua_angle[0]*qua_angle[2]-qua_angle[3]*qua_angle[1]))
        euler_angle[2]=math.atan2(2*(qua_angle[0]*qua_angle[3]+qua_angle[1]*qua_angle[2]),1-2*(qua_angle[2]**2+qua_angle[3]**2))
        return euler_angle
        
def mainloop():
    # Starts the node
    rospy.init_node('att_fus')
    att_pub = rospy.Publisher('/estimated_att', Vector3Stamped)
    i_filter = AttitudeFilter(att_pub)   # Define a variable as AttitudeFilter class
    n=0       
    #Subscribes accelometer, gyro and magnometer data from IMU for our filter
    sub_accel = rospy.Subscriber('/raw_accel',Vector3Stamped, i_filter.acc_update)
    sub_pose = rospy.Subscriber('/raw_gyros', Vector3Stamped, i_filter.gyro_update)
    sub_mag = rospy.Subscriber('/raw_mag', Vector3Stamped, i_filter.mag_update)    
    while not rospy.is_shutdown():
        n=n+1 
        rospy.sleep(0.01)

# Main node function
if __name__ == '__main__':
    try:
        mainloop()
    except rospy.ROSInterruptException: pass
