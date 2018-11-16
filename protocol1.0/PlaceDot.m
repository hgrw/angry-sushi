function PlaceDot(port_num, PROTOCOL_VERSION)
    ADDR_TORQUE_ENABLE       = 24;           % Control table address is different in Dynamixel model
    ADDR_GOAL_POSITION       = 30;
    ADDR_PRESENT_POSITION    = 36;
    ADDR_MOVING_SPEED = 32;
    ADDR_TORQUE_LIMIT = 34;
    
    M3 = 4;

    t = 1;
    
    write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_TORQUE_LIMIT, 70);
    write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 1250);
    
    pause(t);
    
    write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_TORQUE_LIMIT, 800);
    write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 25);
    pause(t);
    write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 0);
end