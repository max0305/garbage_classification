 async def call_tm_driver_set_position(self, motion_type=2, 
     positions=[0.3571, -0.5795, 0.5, -pi, 0., pi/4], velocity=2, acc_time=0.2, 
     blend_percentage=10, fine_goal = True
     ): #default as home position
  req = SetPositions.Request()
  if motion_type in [1,2,4]:
   req.motion_type = motion_type
  else:
   raise ValueError('Motion_type is not allowed: PTP_J = 1, PTP_T = 2, LINE_T = 4')
  req.positions = self.range_chack(motion_type = motion_type, positions = positions) # need a function to check the envelope
  req.velocity = min(velocity,self.tm12_default_setting["velocity_max"])
  req.acc_time = max(acc_time,self.tm12_default_setting["acc_time_min"])
  req.blend_percentage = min(100,max(0,int(blend_percentage+0.5)))
  req.fine_goal = fine_goal
  
  self.get_logger().info(f'Calling tm_driver set_position service')
  future = self.tm12_client.call_async(req)
  await future
  if future.result() is None:
   raise RuntimeError('tm_driver_set_position call failed')
  return future.result()

 async def call_grpr2f85_set(self, position=0, speed=255, force=255, wait_time=0):
  req = SetGripperState.Request()
  req.position = min(255,max(0,int(position+0.5)))
  req.speed = min(255,max(0,int(speed+0.5)))
  req.force = min(255,max(0,int(force+0.5)))
  req.wait_time = max(int(wait_time+0.5),0)
  
  self.get_logger().info(f'Calling my_grpr2f85_set service')
  future = self.gripper2f85_client.call_async(req)
  await future
  if future.result() is None:
   raise RuntimeError('grpr2f85_set call failed')
  return future.result()