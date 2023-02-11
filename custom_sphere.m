
%X = importdata("latlong_data_4week_mark.csv"); % actual values using f
X = importdata("latlong_data_4weeks_testing.csv"); % simplified values using f=0

A = X(2:end, 2:end);

latlong5_6 = A( (500<A(:,3)) & (A(:,3)<=600),:);
latlong6_7 = A( (600<A(:,3)) & (A(:,3)<=700),:);
latlong7_8 = A( (700<A(:,3)) & (A(:,3)<=800),:);
latlong8_9 = A( (800<A(:,3)) & (A(:,3)<=900),:);
latlong9_10 = A( (900<A(:,3)) & (A(:,3)<=1000),:);
latlong10_11 = A( (1000<A(:,3)) & (A(:,3)<=1100),:);
latlong11_12 = A( (1100<A(:,3)) & (A(:,3)<=1200),:);
latlong12_13 = A( (1200<A(:,3)) & (A(:,3)<=1300),:);
latlong13_14 = A( (1300<A(:,3)) & (A(:,3)<=1400),:);

altList = {latlong5_6, latlong6_7, latlong7_8, latlong8_9, latlong9_10,...
    latlong10_11, latlong11_12, latlong12_13, latlong13_14};

title_list = ["500-600km", "600-700km", "700-800km", "800-900km", "900-1000km",...
    "1000-1100km", "1100-1200km", "1200-1300km", "1300-1400km"];

n = 20;

tiledlayout(3,3)
nexttile

for m = 1:1:9
    % see if I can replace this monstrosity elseif statement
    if m==1
        current_alt = latlong5_6;
        titleAlt = "500-600km";
    elseif m==2
        current_alt = latlong6_7;
        titleAlt = "600-700km";
    elseif m==3
        current_alt = latlong7_8;
        titleAlt = "700-800km";
    elseif m==4
        current_alt = latlong8_9;
        titleAlt = "800-900km";
    elseif m==5
        current_alt = latlong9_10;
        titleAlt = "900-1000km";
    elseif m==6
        current_alt = latlong10_11;
        titleAlt = "1000-1100km";
    elseif m==7
        current_alt = latlong11_12;
        titleAlt = "1100-1200km";
    elseif m==8
        current_alt = latlong12_13;
        titleAlt = "1200-1300km";
    elseif m==9
        current_alt = latlong13_14;
        titleAlt = "1300-1400km";
    end

    % end of abomination

    [x,y,z] = sphere(n);
    
    r = 1;
    surf(r.*x,r.*y,r.*z,'FaceColor', 'white', 'FaceAlpha',0); %// Base sphere
    hold on;
    
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
    
%             r_eq = min(1,abs(0.0002057613*histQuant^5 - 0.0077160494*histQuant^4 + 0.0987654321*histQuant^3 - 0.4861111111*histQuant^2 + 0.7611111111*histQuant + 0.0000000000));
%             g_eq = min(1,abs(0.0000857339*histQuant^5 - 0.0033436214*histQuant^4 + 0.0470679012*histQuant^3 - 0.3032407407*histQuant^2 + 0.9027777778*histQuant + 0.0000000000));
%             b_eq = min(1,abs(min(1,abs(0.0001543210*histQuant^5 - 0.0064300412*histQuant^4 + 0.0964506173*histQuant^3 - 0.6087962963*histQuant^2 + 1.2861111111*histQuant + 0.5000000000))));
             
%             r_eq = min(1,abs(0.00000000002346873627*histQuant^5 - 0.00000002153256553152*histQuant^4 + 0.00000674342532268161*histQuant^3 - 0.00081205592126942800*histQuant^2 + 0.03110808355478410000*histQuant - 0.00000000005348255172));
%             g_eq = min(1,abs(0.00000000000977864011*histQuant^5 - 0.00000000933077839646*histQuant^4 + 0.00000321366363021758*histQuant^3 - 0.00050656821755878900*histQuant^2 + 0.03689827429253610000*histQuant - 0.00000000001761346624));
%             b_eq = min(1,abs(0.00000000001760155220*histQuant^5 - 0.00000001794380460856*histQuant^4 + 0.00000658537629150873*histQuant^3 - 0.00101700336808138000*histQuant^2 + 0.05256584922038290000*histQuant + 0.50000000000777400000));

            r_eq = min(1,abs(0.00000017055387002246*histQuant^5 - 0.00002643584985344980*histQuant^4 + 0.00139863269623497000*histQuant^3 - 0.02845343391332020000*histQuant^2 + 0.18413978487075200000*histQuant + 0.00000000015462831016));
            g_eq = min(1,abs(0.00000007106411250492*histQuant^5 - 0.00001145553493593110*histQuant^4 + 0.00066653589428811200*histQuant^3 - 0.01774952306138820000*histQuant^2 + 0.21841397845815900000*histQuant + 0.00000000004359268502));
            b_eq = min(1,abs(0.00000012791540250360*histQuant^5 - 0.00002202987487647530*histQuant^4 + 0.00136585224240340000*histQuant^3 - 0.03563453866520360000*histQuant^2 + 0.31115591390687200000*histQuant + 0.50000000000943400000));

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
    axis equal;
    
    grid on;
    colormap(jet(62));
    colorbar;
    c = colorbar;
    c.FontSize = 12;
    clim([0 62]);
    c.Ticks = [0 12 25 37 50 62];

    names = {'-1'; 'Longitude = -90'; '1'};
    set(gca,'xtick',[-1:1],'xticklabel',names);
    names2 = {'-1'; 'Longitude = 0'; '1'};
    set(gca,'ytick',[-1:1],'yticklabel',names2);

    title(titleAlt);
    view(-50,-35);
    if (m~=9)
        nexttile;
    end

    
end




