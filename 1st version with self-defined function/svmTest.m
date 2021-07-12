function result = svmTest(svm, testData, testLabel)  
 
result.score = svm.w' * testData + svm.b;  
d1Test = sign(result.score);  %f(x)  
result.d1Test = d1Test;  
result.accuracy = size(find(d1Test==testLabel))/size(testLabel);  

end