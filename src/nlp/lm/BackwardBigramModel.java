package nlp.lm;


import java.io.*;
import java.util.*;

/**
 * 
 * @author Huihuang Zheng
 * This class calculates BigramModel but backward
 */

public class BackwardBigramModel extends BigramModel {
	
	/** Accumulate unigram and backward bigram counts for these sentences 
	 *  by reversing sentence
	 */
	@Override
    public void trainSentence (List<String> sentence) {
        ArrayList<String> reverseSentence = new ArrayList<String>(sentence);
        Collections.reverse(reverseSentence);
        super.trainSentence(reverseSentence);
    }

    /** Compute log probability of sentence given current backward model 
     *  Because the train uses backward model, the test also needs to be reversed
     */
	@Override
    public double sentenceLogProb (List<String> sentence) {
    	ArrayList<String> reverseSentence = new ArrayList<String>(sentence);
        Collections.reverse(reverseSentence);
        return super.sentenceLogProb(reverseSentence);
    }

    /** Like sentenceLogProb but excludes predicting start-of-sentence when computing perplexity */
	@Override
    public double sentenceLogProb2 (List<String> sentence) {
    	ArrayList<String> reverseSentence = new ArrayList<String>(sentence);
        Collections.reverse(reverseSentence);
        return super.sentenceLogProb2(reverseSentence);
    }

    /** Returns vector of probabilities of predicting each token in the sentence
     *  including the start of sentence. Using backward order.
     *  
     *  Note: the answer is in backward order to BigramModel
     *  For example:
     *    forwardProbs consists of [(<S>, w1), (w1, w2) ... (w_len, <\S>)];
     *    backwardProbs consists of [(<\s>, w_len), (w_len-1, w_len-2) ... (<w1>, <S>)]
     **/
	
    public double[] sentenceTokenProbs (List<String> sentence) {
    	ArrayList<String> reverseSentence = new ArrayList<String>(sentence);
        Collections.reverse(reverseSentence);
        return super.sentenceTokenProbs(reverseSentence);
    }
	
  
	/** Train and test a bigram model.
     *  Command format: "nlp.lm.BackwardBigramModel [DIR]* [TestFrac]" where DIR 
     *  is the name of a file or directory whose LDC POS Tagged files should be 
     *  used for input data; and TestFrac is the fraction of the sentences
     *  in this data that should be used for testing, the rest for training.
     *  0 < TestFrac < 1
     *  Uses the last fraction of the data for testing and the first part
     *  for training.
     *  
     */

    public static void main(String[] args) throws IOException {
    	// All but last arg is a file/directory of LDC tagged input data
    	File[] files = new File[args.length - 1];
    	for (int i = 0; i < files.length; i++) 
    		files[i] = new File(args[i]);
    	// Last arg is the TestFrac
    	double testFraction = Double.valueOf(args[args.length -1]);
    	// Get list of sentences from the LDC POS tagged input files
    	List<List<String>> sentences = 	POSTaggedFile.convertToTokenLists(files);

    	int numSentences = sentences.size();
    	// Compute number of test sentences based on TestFrac
    	int numTest = (int)Math.round(numSentences * testFraction);
		// Take test sentences from end of data
		List<List<String>> testSentences = sentences.subList(numSentences - numTest, numSentences);
		// Take training sentences from start of data
		List<List<String>> trainSentences = sentences.subList(0, numSentences - numTest);
		System.out.println("# Train Sentences = " + trainSentences.size() + 
		   " (# words = " + wordCount(trainSentences) + 
		   ") \n# Test Sentences = " + testSentences.size() +
		   " (# words = " + wordCount(testSentences) + ")");
		// Create a bigram model and train it.
		BackwardBigramModel model = new BackwardBigramModel();
		System.out.println("Training...");
		model.train(trainSentences);
		// Test on training data using test and test2
		model.test(trainSentences);
		model.test2(trainSentences);
		System.out.println("Testing...");
		// Test on test data using test and test2
		model.test(testSentences);
		model.test2(testSentences);
		
    }
}
