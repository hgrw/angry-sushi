function MoveTo(xb, yb, port_num, PROTOCOL_VERSION)

a1 = 0.365; % m % arm 1 lenght
a2 = 0.205; % m % arm 2 length

M1 = 3;
M2 = 2;

ADDR_TORQUE_ENABLE       = 24;           % Control table address is different in Dynamixel model
ADDR_GOAL_POSITION       = 30;
ADDR_PRESENT_POSITION    = 36;
ADDR_MOVING_SPEED = 32;
ADDR_TORQUE_LIMIT = 34;

% xb = 0.4; % [xb] = m % object world x coord
% yb = 0; % [yb] = m % object world y coord

XB = 0; % [XB]=m % distance between world and servo1 X frame
YB = 0; % [YB]=m % distance between world and servo1 Y frame
rotdeg = 0; % rotation of world frame w.r.t. servo1 frame
rot = degtorad(rotdeg);

x = XB + xb*cos(rot)-yb*sin(rot); % x coord in servo1 frame
y = YB + xb*sin(rot)+yb*cos(rot); % y coord in servo1 frame

t2rad = acos((x^2+y^2-a1^2-a2^2)/(2*a1*a2)); 
t1rad = atan(y/x)-atan((a2*sin(t2rad))/(a1+a2*cos(t2rad)));
t1rad2 = atan2((-x*a2*sin(t2rad)+y*(a1+a2*cos(t2rad))),(x*(a1+a2*cos(t2rad))+y*a2*sin(t2rad)));
q1 = radtodeg(t1rad); % servo 1 angle in deg wrt own frame
q1b = radtodeg(t1rad2); % servo 1 angle in deg wrt own frame (alternative)
q2 = radtodeg(t2rad); % servo 2 angle in deg wrt own frame

deg1 = 150 - q1;       % angle 1 using adjustment
deg1b = 150-q1b;       % angle 1 using adjustment (alternative)
deg2 = 150 - q2;       % angle 2 using adjustment

Motor1_Move(port_num, PROTOCOL_VERSION, deg1);

Motor2_Move(port_num,PROTOCOL_VERSION, deg2); 

m1Pos = 1023 - deg1*1023/300;
m2Pos = 1023 - deg2*1023/300;

m1CurrentPos = read2ByteTxRx(port_num, PROTOCOL_VERSION, M1, ADDR_PRESENT_POSITION);
m2CurrentPos = read2ByteTxRx(port_num, PROTOCOL_VERSION, M2, ADDR_PRESENT_POSITION);

tic;
n = 0;
while 1
    m1CurrentPos = read2ByteTxRx(port_num, PROTOCOL_VERSION, M1, ADDR_PRESENT_POSITION)
    m2CurrentPos = read2ByteTxRx(port_num, PROTOCOL_VERSION, M2, ADDR_PRESENT_POSITION)
    
    if abs(m1CurrentPos-m1Pos)<7 && abs(m2CurrentPos-m2Pos)<7
        break
    end
    n = toc;
    if n>10
        break
    end
end
pause(0.2)
end
