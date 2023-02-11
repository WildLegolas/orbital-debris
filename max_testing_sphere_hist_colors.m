%X = importdata("latlong_data_4week_mark.csv");
X = importdata("latlong_data_4weeks_testing.csv");
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


ls = [];

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
            ls(end+1) = histQuant;
            % color gradient that is universal, 10(or whatever was max from previous
            % quadrilateral and 3d bar hist plots) being top end
            
            % manual interpolation
           

            

            
        end
    end
    

    
end

max(ls)
size_A = size(A,1)
sum_range = sum( ls , 'all' )
coverage_range = 100*sum_range/size_A;
fprintf('The 500-1400km range covers %.6f percent of all debris', coverage_range)