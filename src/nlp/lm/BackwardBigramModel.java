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
	
	/**  Backward Bigram model that maps a backword bigram "A B" as a string "B\nA" to the
     *   P(A | B) 
     *   public Map<String, DoubleValue> bigramMap;
	 */
	/** Accumulate unigram and backward bigram counts for this sentence */
	
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
	}
	
	
	/* Compute log probability of sentence given current model */
    public double sentenceLogProb (List<String> sentence) {
    	// Set start-sentence as initial token
    	String lastToken = "</S>";
    	// Maintain total sentence prob as sum of individual token
    	// log probs (since adding logs is same as multiplying probs)
    	double sentenceLogProb = 0;
    	// Check prediction of each token in sentence
    	for (int i = sentence.size() - 1; i >= 0;--i) {
    		String token = sentence.get(i);
    		// Retrieve unigram prob
    		DoubleValue unigramVal = unigramMap.get(token);
    		if (unigramVal == null) {
    			// If token not in unigram model, treat as <UNK> token
    			token = "<UNK>";
    			unigramVal = unigramMap.get(token);
    		}
    		// Get bigram prob
    		String bigram = bigram(lastToken, token);
    		DoubleValue bigramVal = bigramMap.get(bigram);
    		// Compute log prob of token using interpolated prob of unigram and bigram
    		double logProb = Math.log(interpolatedProb(unigramVal, bigramVal));
    		// Add token log prob to sentence log prob
    		sentenceLogProb += logProb;
    		// update previous token and move to next token
    		lastToken = token;
    	}
    	// Check prediction of end of sentence token
    	DoubleValue unigramVal = unigramMap.get("<S>");
    	String bigram = bigram(lastToken, "<S>");
    	DoubleValue bigramVal = bigramMap.get(bigram);
    	double logProb = Math.log(interpolatedProb(unigramVal, bigramVal));
    	// Update sentence log prob based on prediction of </S>
    	sentenceLogProb += logProb;
    	return sentenceLogProb;
    }
    
    /** Like sentenceLogProb but excludes predicting end-of-sentence when computing prob */
    public double sentenceLogProb2 (List<String> sentence) {
    	String lastToken = "</S>";
    	double sentenceLogProb = 0;
    	for (int i = sentence.size() - 1; i >= 0; --i) {
    		String token = sentence.get(i);
    		DoubleValue unigramVal = unigramMap.get(token);
    		if (unigramVal == null) {
    			token = "<UNK>";
    			unigramVal = unigramMap.get(token);
    		}
    		String bigram = bigram(lastToken, token);
    		DoubleValue bigramVal = bigramMap.get(bigram);
    		double logProb = Math.log(interpolatedProb(unigramVal, bigramVal));
    		sentenceLogProb += logProb;
    		lastToken = token;
    	}
    	return sentenceLogProb;
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
