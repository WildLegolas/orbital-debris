
X = importdata("latlong_data.csv");
A = X(2:end, 2:end);


latlong7_8 = A( (700<A(:,3)) & (A(:,3)<=800),:);
latlong8_9 = A( (800<A(:,3)) & (A(:,3)<=900),:);
latlong9_10 = A( (900<A(:,3)) & (A(:,3)<=1000),:);

n = 20;


    [x,y,z] = sphere(n);
    
    r = 1;
    surf(r.*x,r.*y,r.*z,'FaceColor', 'white', 'FaceAlpha',0); %// Base sphere
    hold on;
    current_alt = latlong7_8;
    for countAz = 1:1:n
        for countE = 1:1:n
            
    
            latMin = -90 + (countE-1)*(180/n);
            latMax = -90 + (countE)*(180/n);
            longMin = -180 + (countAz-1)*(360/n);
            longMax = -180 + (countAz)*(360/n);
    
            % get rows that fit criteria of being within current latitude and
            % longitude ranges
            currentCheck = current_alt( (latMin<=current_alt(:,1)) & (latMax>=current_alt(:,1)) ...
                & (longMin<=current_alt(:,2)) & (longMax>=current_alt(:,2)) ,:);
    
            histQuant = size(currentCheck,1); % number of rows within the latitude and longitude ranges for given altitude range
            
            % color gradient that is universal, 10(or whatever was max from previous
            % quadrilateral and 3d bar hist plots) being top end
            
            % manual interpolation
            max_val = 25;
            
            r_eq = min(1,abs(0.0002057613*histQuant^5 - 0.0077160494*histQuant^4 + 0.0987654321*histQuant^3 - 0.4861111111*histQuant^2 + 0.7611111111*histQuant + 0.0000000000));
            g_eq = min(1,abs(0.0000857339*histQuant^5 - 0.0033436214*histQuant^4 + 0.0470679012*histQuant^3 - 0.3032407407*histQuant^2 + 0.9027777778*histQuant + 0.0000000000));
            b_eq = min(1,abs(min(1,abs(0.0001543210*histQuant^5 - 0.0064300412*histQuant^4 + 0.0964506173*histQuant^3 - 0.6087962963*histQuant^2 + 1.2861111111*histQuant + 0.5000000000))));


            hist_color = [r_eq, g_eq, b_eq];
            
    
    
            % Get a 2x2 patch of points. Each row of the points is an elevation. Points are the intersection of the black grid.
            x2=[x(countE,countAz),x(countE,countAz+1),x(countE+1,countAz),x(countE+1,countAz+1)];
            y2=[y(countE,countAz),y(countE,countAz+1),y(countE+1,countAz),y(countE+1,countAz+1)];
            z2=[z(countE,countAz),z(countE,countAz+1),z(countE+1,countAz),z(countE+1,countAz+1)];
            x2 = reshape(x2,2,2);
            y2 = reshape(y2,2,2);
            z2 = reshape(z2,2,2);
    
                            
            random_color = rand(1,3);
            surf(r*x2,r*y2,r*z2,'FaceColor',hist_color, 'FaceAlpha',1);

            
        end
    end
    
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    quiver3(0,-1,0,0,-0.3,0, 'red');
    quiver3(1,0,0,0.3,0,0, 'red');
    quiver3(0,0,1,0,0,0.3, 'red');
    grid on;
colormap(jet(15));
colorbar;
c = colorbar;
c.FontSize = 12;
caxis([0 15]);
c.Ticks = [0 3 6 9 12 15];

names = {'-1'; 'Longitude = -90'; '1'};
set(gca,'xtick',[-1:1],'xticklabel',names);
names2 = {'-1'; 'Longitude = 0'; '1'};
set(gca,'ytick',[-1:1],'yticklabel',names2);

    axis equal;
    view(-50,-35);


    
