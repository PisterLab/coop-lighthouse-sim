import numpy as np
import math
import matplotlib.pyplot as plt


timesteps = 50
dt = 0.5
# timesteps per rotation
sample_rate = 10
# covariance of lighthouse states
P_l = np.identity(3)

x_l0 = 0
y_l0 = 0

iterations = 1000
plot_run = False
error = np.empty((0))

for iteration in range(iterations):
    x_l_traj = 0 * np.cos(np.linspace(0, timesteps, num=timesteps)/100)
    y_l_traj = 3 * np.sin(dt*np.linspace(0, timesteps, num=timesteps))
    #y_l_traj = 2 * (unidrnd(2*ones(1,timesteps))-1.5);
    #x0 = 3
    #y0 = 0
    
    x0 = np.random.rand() * 5 - 2.5
    y0 = np.random.rand() * 5 - 2.5
    
    # x0 = 3.4080;
    # y0= 1.888;
    varx = [np.power(.3, 2)]
    vary = [np.power(.3, 2)]

    x_a = np.array([x0, y0])[:, None]
    P_m = np.empty((2, 2, 0))
    P_m = np.append(P_m, np.array([[varx[0], 0], [0, vary[0]]])[:,:,None], axis=2)
    # initial p_m
    x_m = np.array([x0 + np.random.randn() * np.sqrt(varx[0]), y0 + np.random.randn() * np.sqrt(vary[0])])[:, None]
    K_rx = [0]
    K_ry = [0]
    K_lx = [0]
    K_ly = [0]
    measurement = np.zeros((2, 1))

    # noise stds
    sig1 = .05
    sig2 = .05
    sig3 = 1.5 * 3.1415 / 180
    sig4 = 10

    P_l = np.diag([sig1**2, sig2**2, sig3**2]) # covariance of lighthouse states 

    D, V = np.linalg.eig(P_m[:,:,0])
    D = np.diag(D)[:,:,None]
    V = V[:,:,None]

    
    # control vectors for lighthouse
    theta = 3.14 / 8
    rot = np.array([[np.cos(theta), - np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    u_primative = np.array([[1, 0, 0, -1], 
                            [0, 1, -1, 0]])
    # u_l = [1,0;0,1;0,-1;-1,0;1,1;1,-1;-1,1;-1,-1]' * 1;
    u_l = np.concatenate((u_primative, u_primative * 0.5, rot @ u_primative, rot @ u_primative * .5, rot @ rot @ u_primative, rot @ rot @ u_primative * .5), axis=1)
    x_l = [0]
    y_l = [0]
    X_l = np.array([x_l[0], y_l[0]])[:, None]
    last_direction = np.array([0, 0])[:, None]
    r_diffx = []
    r_diffy = []
    x_p = np.zeros((2, 1))
    P_p = np.zeros((2,2,1))

    # Begin for loop 
    for i in range(1, timesteps):
        # step dynamics forward 
        x_a = np.append(x_a, x_a[:, i-1][:, None], axis=1)
        
        max_idx = 1
        # lighthouse location control 

        #WHEN IS THIS SUPPOSED TO GO INTO THIS IF STATEMENT
        if i < 1:
            gain = np.array([]) 
            for cont in range(0, 9):
                Rp = np.diag([np.power(sig1, 2), np.power(sig4, 2)])
                del_Xl_prop = u_l[:, cont]
                del_x = del_Xl_prop[0]
                del_y = del_Xl_prop[1]
                d = np.linalg.norm(x_m[:, i-1]-(X_l[:,i-1] + del_Xl_prop))
                angle = np.arctan2(x_m[1,i-1]-(y_l[i-1]+del_y), x_m[0,i-1]-(x_l[i-1]+del_x))
                # Hp = [-(x_m(2,i-1)-(y_l(i-1)+del_y))/norm(x_m(:,i-1)-(X_l(:,i-1)+del_Xl_prop))^2 , (x_m(1,i-1)-(x_l(i-1)+del_x))/norm(x_m(:,i-1)-(X_l(:,i-1)+del_Xl_prop))^2;
                # -10*(x_m(1,i-1)-(x_l(i-1)+del_x))/(log(10)* norm(x_m(:,i-1)-(X_l(:,i-1) + del_Xl_prop))^2), -10*(x_m(2,i-1)-(y_l(i-1) + del_y))/(log(10)* norm(x_m(:,i-1)-(X_l(:,i-1)+del_Xl_prop))^2)];
                Hp = (1/d) * np.array([[np.sin(angle) , -np.cos(angle)]])
                # 10*(x_m(1,i-1)-(x_l(i-1)+del_x))/(log(10)* d), 10*(x_m(2,i-1)-(y_l(i-1) + del_y))/(log(10)* d)];
                Rp = np.linalg.inv(Rp)
                # using least squares here instead of matrix right division
                fim = np.matmul(Hp.T / np.power(sig1, 2), Hp)
                area = np.linalg.det(fim)
                d, v = np.linalg.eig(fim)
                d = np.diag(d)
                gain = np.append(gain, area)
                # gain(cont) = d(1,1);

            argvalue = np.max(gain)
            max_idx = np.argmax(gain)
            d = np.linalg.norm(x_m[:,i-1] - X_l[:,i-1])
            angle = np.arctan2(x_m[1,i-1] - y_l[i-1], x_m[0,i-1] - x_l[i-1])
            Hp = (1/d) * np.array([[np.sin(angle), -np.cos(angle)]])
            fim = np.matmul(Hp.T / np.power(sig1,2), Hp)
            d, v = np.linalg.eig(fim)
            d = np.diag(d)
            if d[0, 0] > d[1, 1]:
                direction = v[:, 0]
            else:
                direction = v[:, 1]

            last_direction = direction
        else:
            d = np.linalg.norm(x_m[:,i-1] - X_l[:,i-1])
            angle = np.arctan2(x_m[1, i-1] - y_l[i-1], x_m[0, i-1] - x_l[i-1])
            Hp = (1/d) * np.array([[np.sin(angle), -np.cos(angle)]])
            # -10*(x_m(1,i-1)-(x_l(i-1)))/(log(10)* d), -10*(x_m(2,i-1)-(y_l(i-1)))/(log(10)* d)];
            Rp = np.diag([np.power(sig1, 2)])
            fim = np.matmul(Hp.T / np.linalg.inv(Rp), Hp)
            lam, v = np.linalg.eig(fim)
            lam = np.diag(lam)
            if lam[0, 0] >= lam[1, 1]:
                direction = v[:, 0]
            else:
                direction = v[:, 1]
            
            dot = np.matmul(np.transpose(direction), last_direction)
            
            if (np.matmul(np.transpose(direction), last_direction)) < 0:
                direction = -direction

            last_direction = direction

        # x_l(i) = x_l(i-1)+ u_l(1,max_idx);
        # y_l(i) = y_l(i-1)+ u_l(2,max_idx);
        x_l = np.append(x_l, x_l_traj[i])
        y_l = np.append(y_l, y_l_traj[i])
        
        x_l[i] = x_l[i-1] + direction[0]
        y_l[i] = y_l[i-1] + direction[1]
        X_l = np.append(X_l, np.array([x_l[i], y_l[i]])[:,None], axis=1)
        # prediction step
        x_p = np.append(x_p, x_m[:, i-1][:, None], axis=1)
        P_p = np.append(P_p, P_m[:, :, i-1][:,:,None], axis=2)

        # generate noise
        w1 = np.random.randn() * sig1
        w2 = np.random.randn() * sig2
        w3 = np.random.randn() * sig3
        # w4 = (randn(1) * sig4);
        # w4 = sig4+max(-exprnd(sig4),-90);
        w4 = -np.random.rayleigh(sig4 / np.sqrt((4-3.14)/2)) # rayleigh fading
        # generate measurments 
        z = np.array([[np.arctan2(x_a[1,i] - (y_l[i] + w1), x_a[0, i] - (x_l[i] + w2)) + w3],
            [-10 * np.log10(np.linalg.norm(x_a[:,i] - np.array([[x_l[i]], [y_l[i]]]))) + w4]])
        # propogate prediction through measurment model
        # z
        h = np.array([[np.arctan2(x_p[1, i] - y_l[i], x_p[0, i] - x_l[i])],
                        [-10 * np.log10(np.linalg.norm(x_p[:, i] - X_l[:, i]))]])
        # measurement step

        if abs(z[0] - h[0]) < 3.14:
            r = np.linalg.norm(x_p[:, i] - X_l[:, i])
            angle = np.arctan2(x_p[1, i] - y_l[i], x_p[0, i] - x_l[i])
            H = (1/r) * np.array([[-np.sin(angle), np.cos(angle)],
                [-10 * (x_p[0, i] - x_l[i]) / (np.log(10) * r), -10 * (x_p[1, i] - y_l[i]) / (np.log(10) * r)]])
            # H = [-(x_p(2,i)-y_l(i))/norm(x_p(:,i)-X_l(:,i))^2 , (x_p(1,i)-x_l(i))/norm(x_p(:,i)-X_l(:,i))^2;
            #      -10*(x_p(1,i)-x_l(i))/(log(10)* norm(x_p(:,i)-X_l(:,i))^2), -10*(x_p(2,i)-y_l(i))/(log(10)* norm(x_p(:,i)-X_l(:,i))^2)];


            W = np.array([[(x_p[1, i] - y_l[i]) / np.power(np.linalg.norm(x_p[:, i] - X_l[:, i]), 2), -(x_p[0, i] - x_l[i]) / np.power(np.linalg.norm(x_p[:, i] - X_l[:, i]), 2), 1, 0],
                            [10 * (x_p[0, i] - x_l[i]) / (np.log(10) * np.power(np.linalg.norm(x_p[:, i] - X_l[:, i]), 2)), 10*(x_p[1, i]-y_l[i]) / (np.log(10) * np.power(np.linalg.norm(x_p[:, i] - X_l[:, i]), 2)), 0 ,1]])

            R = np.array([np.append(P_l[0,:],[0]),
                        np.append(P_l[1,:], [0]),
                        np.append(P_l[2,:], [0]),
                        [0,0,0,sig4**2]])



            K = P_p[:,:,i] @ H.T @ np.linalg.inv(H @ P_p[:,:,i] @ H.T + W @ R @ W.T)

            # is the kalman gain helpful?

            K = np.array([[K[0, 0], 0],
                        [K[1, 0], 0]])
            # K = [0,K(1,2); 0,K(2,2)];



            # K*H

            # K;
            # H;
            # z-h;
            # z;

            x_m = np.append(x_m, np.array(x_p[:, i][:,None] + K @ (z-h)), axis=1)
            P_m = np.append(P_m, np.array((np.identity(2) - K @ H) @ P_p[:,:,i])[:,:,None], axis=2)
            varx = np.append(varx, P_m[0,0,i])
            vary = np.append(vary, P_m[1,1,i])
            measurement = np.append(measurement, z, axis=1)

            K_rx = np.append(K_rx, K[0,1])
            K_ry = np.append(K_ry, K[1,1])
            K_lx = np.append(K_lx, K[0,0])
            K_ly = np.append(K_ly, K[1,0])

            r_diffx = np.append(r_diffx, K[0,1]*(z[1]-h[1]))
            r_diffy = np.append(r_diffy, K[1,1]*(z[1]-h[1]))
            tempD, tempV = np.linalg.eig(P_m[:,:,i])
            V = np.append(V, tempV)
            D = np.append(D, np.diag(tempD))
        else:
            x_m = np.append(x_m, x_m[:,i-1][:,None], axis=1)
            P_m = np.append(P_m, P_m[:,:,i-1][:,:,None], axis=2)
            varx = np.append(varx, P_m[0,0,i])
            vary = np.append(vary, P_m[1,1,i])
            # varx(i) = D(1,1,i);
            # vary(i) = D(2,2,i);
            measurement = np.append(measurement, z, axis=1)
            K_rx = np.append(K_rx, K_rx[i-1])
            K_ry = np.append(K_ry, K_ry[i-1])
            r_diffx = np.append(r_diffx, 0)
            r_diffy = np.append(r_diffy, 0)
            tempD, tempV = np.linalg.eig(P_m[:,:,i])
            V = np.append(V, tempV)
            D = np.append(D, np.diag(tempD))

    error = np.append(error, np.linalg.norm(x_a[:,0]-x_m[:,-1]))
    
    # Plot Runs NOT ORIGINALLY COMMENTED OUT BUT PLOT RUN IS FALSE SO TESTING
    # if plot_run:
    #     linewidth = 2;
    #     x_m(:,timesteps)
    #     P_m(:,:,timesteps)
    #     plot([1:timesteps],x_m(1,:),[1:timesteps],x_m(2,:))
    #      set(findall(gca, 'Type', 'Line'),'LineWidth',linewidth);
    #      title('Estimated Anchor Location')

    #      xlabel('Measurement Number')
    #      ylabel('Location (m)')
    #      set(gca,'fontsize',20)
    #      hold

    #      plot([1:timesteps],ones(1,timesteps)*x_a(1,1),'--b',[1:timesteps],ones(1,timesteps)*x_a(2,2),'--r')
    #      legend('X','Y','X truth','Y truth')
    #      xlim([0,100])
    #     figure

    #     subplot(1,2,1)
    #     plot([1:timesteps],abs(x_m(1,:)-x_a(1,1)))
    #      set(findall(gca, 'Type', 'Line'),'LineWidth',linewidth);
    #      title('X Location Error')
    #      set(gca,'fontsize',20)
    #      xlabel('Measurement Number')
    #      ylabel('Error (m)')
    #      set(gca,'YScale','log')
    #     xlim([0,100])

    #      subplot(1,2,2)
    #      plot([1:timesteps],abs(x_m(2,:)-x_a(2,1)))
    #      set(findall(gca, 'Type', 'Line'),'LineWidth',linewidth);
    #      title('Y Location Error')
    #      xlabel('Measurement Number')
    #      ylabel('Error (m)')
    #      set(gca,'fontsize',20)
    #      set(gca,'YScale','log')
    #     xlim([0,100])

    #     figure

    #     plot([1:timesteps],varx,[1:timesteps],vary)
    #      set(findall(gca, 'Type', 'Line'),'LineWidth',linewidth);
    #      title('Anchor Location Variance')
    #      legend('Axis 1','Axis 2')
    #      xlabel('Measurement Number')
    #      ylabel('Location (m)')
    #      set(gca,'YScale','log')
    #      set(gca,'fontsize',20)
    #      xlim([0,100])


    #     % %ylim([0,100])
    #     % xlim([0,100])
    #     % 
    #     figure
    #     plot([2:timesteps],abs(K_rx(2:end)),[2:timesteps],abs(K_ry(2:end)))
    #      set(findall(gca, 'Type', 'Line'),'LineWidth',linewidth);
    #       title('Kalman Gain of RSSI Measurements')
    #      legend('X Gain','Y Gain')
    #      xlabel('Measurement Number')
    #      ylabel('Gain')
    #      set(gca,'YScale','log')
    #      set(gca,'fontsize',20)
    #      xlim([0, 100])

    #      figure
    #      plot([1:timesteps],r_diffx,[1:timesteps],r_diffy)
    #       set(findall(gca, 'Type', 'Line'),'LineWidth',linewidth);
    #       title('State Correction of RSSI Measurements')
    #      legend('X','Y')
    #      xlabel('Measurement Number')
    #      ylabel('Location (m)')
    #      set(gca,'fontsize',20)

    #     figure
    #     plot([1:timesteps],y_l)
    #      set(findall(gca, 'Type', 'Line'),'LineWidth',linewidth);
    #       title('Lighthouse Y Location')
    #      xlabel('Measurement Number')
    #      ylabel('Location (m)')
    #      set(gca,'fontsize',20)
    #      xlim([0, 100])
         
    #       figure
    #     plot([1:timesteps],x_l)
    #      set(findall(gca, 'Type', 'Line'),'LineWidth',linewidth);
    #       title('Lighthouse X Location')
    #      xlabel('Measurement Number')
    #      ylabel('Location (m)')
    #      set(gca,'fontsize',20)
    #      xlim([0, 100])

     
        
    #      endpoint = 10;
    #      figure
    #      plot([1:timesteps],measurement(1,:))
    #       set(findall(gca, 'Type', 'Line'),'LineWidth',linewidth);
    #      figure
    #      scatter(x_l(1:endpoint),y_l(1:endpoint))
    #       set(findall(gca, 'Type', 'Line'),'LineWidth',linewidth);
    #       legend('Actual Lighthouse Location')
    #      hold

    #      scatter(x_m(1,1:endpoint),x_m(2,1:endpoint))
    #      legend('Estimated Anchor Position')
    #      scatter(x_a(1,1), x_a(2,1))
    #      legend('Actual Anchor Position')
    #      %hold
    #      a = [1:endpoint]'; b = num2str(a); c = cellstr(b);
    #     dx = 0.1; dy = 0.1; % displacement so the text does not overlay the data points
    #     text(x_m(1,1:endpoint)+dx, x_m(2,1:endpoint)+dy, c);

    #     text(x_l(1:endpoint)+dx, y_l(1:endpoint)+dy, c);
    #       title('Anchor and Lighthouse Location')

    #      xlabel('X (m)')
    #      ylabel('Y (m)')
    #      set(gca,'fontsize',20)
    #      xlim([-6, 6])
    #      ylim([-6, 6])

     
'''
%          figure
%          plot([2:timesteps],measurement(2,2:end))
%           set(findall(gca, 'Type', 'Line'),'LineWidth',linewidth);
%           title('RSSI Measurements')
%          xlabel('Measurement Number')
%          ylabel('RSSI')
%          set(gca,'fontsize',20)
%          xlim([0, 100])
% 
%          figure
%          linewidth = 4;
%          lims = 1
%          subplot(2,2,1)
%          vectors = V(:,:,1)*sqrtm(D(:,:,1))
%          plotv([vectors,-1*vectors])
%          xlim([-lims,lims])
%          ylim([-lims,lims])
%          set(findall(gca, 'Type', 'Line'),'LineWidth',linewidth);
%            title('Measurement 1 ')
%          xlabel('X (m)')
%          ylabel('Y (m)')
%          set(gca,'fontsize',20)
% 
% 
%           subplot(2,2,2)
%          vectors = V(:,:,2)*sqrtm(D(:,:,2))
%          plotv([vectors,-1*vectors])
%          xlim([-lims,lims])
%          ylim([-lims,lims])
%          set(findall(gca, 'Type', 'Line'),'LineWidth',linewidth);
%             title('Measurement 2 ')
%          xlabel('X (m)')
%          ylabel('Y (m)')
%          set(gca,'fontsize',20)
% 
%           subplot(2,2,3)
%          vectors = V(:,:,3)*sqrtm(D(:,:,3))
%          plotv([vectors,-1*vectors])
%          xlim([-lims,lims])
%          ylim([-lims,lims])
%          set(findall(gca, 'Type', 'Line'),'LineWidth',linewidth);
%             title('Measurement 3 ')
%          xlabel('X (m)')
%          ylabel('Y (m)')
%          set(gca,'fontsize',20)
% 
%           subplot(2,2,4)
%          vectors = V(:,:,4)*sqrtm(D(:,:,4))
%          plotv([vectors,-1*vectors])
%          xlim([-lims,lims])
%          ylim([-lims,lims])
%         set(findall(gca, 'Type', 'Line'),'LineWidth',linewidth);
%            title('Measurement 4 ')
%          xlabel('X (m)')
%          ylabel('Y (m)')
%          set(gca,'fontsize',20)
    end
'''


qs, counts = np.unique(error, return_counts=True)
cumulative_prob = np.cumsum(counts).astype(np.double) / error.size

plt.figure(1)

plt.hist(error,100)
print(np.std(error))
print(np.mean(error))
print(np.median(error))
plt.title('Error After 50 Measurements', fontsize = 20)
plt.xlabel('L2 Norm Error (m)', fontsize = 16)
plt.ylabel('Count', fontsize=16)

plt.figure(2)
plt.plot(qs, cumulative_prob)
plt.title('Error After 50 Measurements', fontsize=20)
plt.xlabel('L2 Norm Error (m)', fontsize=16)
plt.ylabel('CDF', fontsize=16)

plt.show()
