
echo "---------- compiling ------------"
javac -d bin -sourcepath src src/nlp/lm/*.java

echo "---------- running --------------"

echo ""
echo "----------BigramModel:-----------"
echo ""
echo "----atis:----"
java -cp bin nlp.lm.BigramModel ./PartOfSpeechTaggedData/atis/ 0.1 
echo ""
echo "----wsj:----"
java -cp bin nlp.lm.BigramModel ./PartOfSpeechTaggedData/wsj/ 0.1 
echo ""
echo "----brown:----"
java -cp bin nlp.lm.BigramModel ./PartOfSpeechTaggedData/brown/ 0.1 

echo ""
echo "-------------BackwardBigramModel:-------------"
echo ""
echo "----atis:----"
java -cp bin nlp.lm.BackwardBigramModel ./PartOfSpeechTaggedData/atis/ 0.1 
echo ""
echo "----wsj:----"
java -cp bin nlp.lm.BackwardBigramModel ./PartOfSpeechTaggedData/wsj/ 0.1 
echo ""
echo "----brown:----"
java -cp bin nlp.lm.BackwardBigramModel ./PartOfSpeechTaggedData/brown/ 0.1 

echo ""
echo "----------BidirectionalBigramModel:-------------"
echo ""
echo "----atis:----"
java -cp bin nlp.lm.BidirectionalBigramModel ./PartOfSpeechTaggedData/atis/ 0.1 
echo ""
echo "----wsj:----"
java -cp bin nlp.lm.BidirectionalBigramModel ./PartOfSpeechTaggedData/wsj/ 0.1 
echo ""
echo "----brown:----"
java -cp bin nlp.lm.BidirectionalBigramModel ./PartOfSpeechTaggedData/brown/ 0.1

