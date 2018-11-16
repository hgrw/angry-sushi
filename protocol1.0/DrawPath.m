function DrawPath(port_num, PROTOCOL_VERSION, path_X, path_Y)

    ADDR_TORQUE_ENABLE       = 24;           % Control table address is different in Dynamixel model
    ADDR_GOAL_POSITION       = 30;
    ADDR_PRESENT_POSITION    = 36;
    ADDR_MOVING_SPEED = 32;
    ADDR_TORQUE_LIMIT = 34;
    
    M3 = 4;
    
    t = 0.2;
    
%     write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_TORQUE_LIMIT, 800);
%     write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 40);             
        
    for i = 1:(length(path_X)-1)   
        
        MoveTo(path_X(i),path_Y(i),port_num, PROTOCOL_VERSION); 
                       
        pause(1);
        
        write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_TORQUE_LIMIT, 70);
        write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 1200);
        
        pause(0.5);
        
        write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_TORQUE_LIMIT, 800);
        write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 40);
        
        pause(0.5)
        write2ByteTxRx(port_num, PROTOCOL_VERSION, M3, ADDR_MOVING_SPEED, 0);
        
    end
end 
        
        
        