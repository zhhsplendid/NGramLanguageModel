package nlp.lm;


import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 
 * @author Huihuang Zheng
 * This class calculates BigramModel but backward
 */

public class BackwardBigramModel extends BigramModel {
	
	/**  Backward Bigram model that maps a backword bigram as a string "A\nB" to the
     *   P(A | B) 
     *   public Map<String, DoubleValue> bigramMap;
	 */
	
	/** Accumulate unigram and backward bigram counts for this sentence */
	/*
	@Override
	public void trainSentence (List<String> sentence) {
		// First count an initial start sentence token
    	String lastToken = "</S>";
    	DoubleValue unigramValue = unigramMap.get("</S>");
    	unigramValue.increment();
    	tokenCount++;
    	// For each token in sentence, accumulate a unigram and bigram count
    	for (int i = sentence.size() - 1; i >= 0; --i) {
    		String token = sentence.get(i);
    		unigramValue = unigramMap.get(token);
    		// If this is the first time token is seen then count it
    		// as an unkown token (<UNK>) to handle out-of-vocabulary 
    		// items in testing
    		if (unigramValue == null) {
    			// Store token in unigram map with 0 count to indicate that
    			// token has been seen but not counted
    			unigramMap.put(token, new DoubleValue());
    			token = "<UNK>";
    			unigramValue = unigramMap.get(token);
    		}
    		unigramValue.increment();    // Count unigram
    		tokenCount++;               // Count token
    		// Make bigram string 
    		String bigram = bigram(lastToken, token);
    		DoubleValue bigramValue = bigramMap.get(bigram);
    		if (bigramValue == null) {
    			// If previously unseen bigram, then
    			// initialize it with a value
    			bigramValue = new DoubleValue();
    			bigramMap.put(bigram, bigramValue);
    		}
    		// Count bigram
    		bigramValue.increment();
    		lastToken = token;
    	}
    	// Account for end of sentence unigram
    	unigramValue = unigramMap.get("<S>");
    	unigramValue.increment();
    	tokenCount++;
    	// Account for end of sentence bigram
    	String bigram = bigram(lastToken, "<S>");
    	DoubleValue bigramValue = bigramMap.get(bigram);
    	if (bigramValue == null) {
    		bigramValue = new DoubleValue();
    		bigramMap.put(bigram, bigramValue);
    	}
    	bigramValue.increment();
	}*/
	
	
	/** Compute unigram and bigram probabilities from unigram and bigram counts */
	@Override
    public void calculateProbs() {
    	// Set bigram values to conditional probability of second token given first
    	for (Map.Entry<String, DoubleValue> entry : bigramMap.entrySet()) {
    		// An entry in the HashMap maps a token to a DoubleValue
    		String bigram = entry.getKey();
    		// The value for the token is in the value of the DoubleValue
    		DoubleValue value = entry.getValue();
    		double bigramCount = value.getValue();
    		String token2 = bigramToken2(bigram); // Get second token of bigram
    		// Prob is ratio of bigram count to token2 unigram count
    		double condProb = bigramCount / unigramMap.get(token2).getValue();
    		// Set map value to conditional probability 
    		value.setValue(condProb);
    	}
    	// Store unigrams with zero count to remove from map
    	List<String> zeroTokens = new ArrayList<String>();
    	// Set unigram values to unigram probability
    	for (Map.Entry<String, DoubleValue> entry : unigramMap.entrySet()) {
    		// An entry in the HashMap maps a token to a DoubleValue
    		String token = entry.getKey();
    		// Uniggram count is the current map value
    		DoubleValue value = entry.getValue();
    		double count = value.getValue();
    		if (count == 0) 
    			// If count is zero (due to first encounter as <UNK>)
    			// then remove save it to remove from map
    			zeroTokens.add(token);
    		else
    			// Set map value to prob of unigram
    			value.setValue(count / tokenCount);
    	}
    	// Remove zero count unigrams from map
    	for (String token : zeroTokens) 
    		unigramMap.remove(token);
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
     *  The same as BigramModel.java
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
		BigramModel model = new BigramModel();
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
