
echo "---------- compiling ------------"
javac -d bin -sourcepath src/nlp/lm/*.java

echo "---------- running --------------"
java -cp bin nlp.lm.BigramModel ./PartOfSpeechTaggedData/atis/ 0.1 
java -cp bin nlp.lm.BigramModel ./PartOfSpeechTaggedData/wsj/ 0.1 
java -cp bin nlp.lm.BigramModel ./PartOfSpeechTaggedData/brown/ 0.1 

