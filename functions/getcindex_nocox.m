function[cindex]=getcindex_nocox(predict,times,status) 
%a function to return the Harrell's cindex(c-statistic) for the cox model

sum1=0;
sum2=0;
sum3=0;
sum4=0;

for i=1:1:(size(predict,1)-1)
    for j=i+1:1:size(predict,1)
        stime1=times(i);
        stime2=times(j);
        pred1=predict(i);
        pred2=predict(j);
        status1=status(i);
        status2=status(j);
        if stime1<stime2 && pred1<pred2 && status1==1
            sum1=sum1+1;
        end
        if stime2<stime1 && pred2<pred1 && status2==1
             sum2=sum2+1;
        end
        if stime1<stime2 && status1==1
            sum3=sum3+1;
        end
        if stime2<stime1 && status2==1
            sum4=sum4+1;
        end
    end
end
cindex=(sum1+sum2)/(sum3+sum4);
end