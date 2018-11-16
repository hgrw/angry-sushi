function DrawDots(port_num, PROTOCOL_VERSION)
ADDR_TORQUE_ENABLE       = 24;           % Control table address is different in Dynamixel model
ADDR_GOAL_POSITION       = 30;
ADDR_PRESENT_POSITION    = 36;
ADDR_MOVING_SPEED = 32;
ADDR_TORQUE_LIMIT = 34;

t_interval = 0.2;
n_dots = 10;

    MoveTo(0.4,-0.1,port_num, PROTOCOL_VERSION);
    pause(4);
    M3 = 4;
    write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_TORQUE_LIMIT, 70);
    write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 1200);
    pause(3);
    MoveTo(0.4,0.1,port_num, PROTOCOL_VERSION);
    
    n = 0;
    while n < n_dots
        write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_TORQUE_LIMIT, 800);
        write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 20);
        pause(t_interval);
        write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_TORQUE_LIMIT, 70);
        write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 1200);
        pause(t_interval);
        n = n + 1;
    end

    
    write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_TORQUE_LIMIT, 800);
    write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 20);
    pause(1);
    write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 0);
end