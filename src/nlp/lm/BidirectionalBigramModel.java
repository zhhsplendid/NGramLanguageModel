package nlp.lm;


import java.io.*;
import java.util.*;

/**
 * 
 * @author Huihuang Zheng
 * 
 */
public class BidirectionalBigramModel extends BackwardBigramModel{
	
	BigramModel bigramModel;
    BackwardBigramModel backwardBigramModel;
    
    /** Interpolation weight for bigram model */
    public double forwardWeight = 0.5;

    /** Interpolation weight for backward model */
    public double backwardWeight = 0.5;
    
    public BidirectionalBigramModel() {
        bigramModel = new BigramModel();
        backwardBigramModel = new BackwardBigramModel();
    }
    
    /**
     * Train two models seperately
     */
    @Override
    public void train (List<List<String>> sentences) {
        bigramModel.train(sentences);
        backwardBigramModel.train(sentences);
    }
    
    /**
     * Compute log probability of sentence given current model
     * Include <S> and </S>. Average these two for boundary prediction
     */
    @Override
    public double sentenceLogProb (List<String> sentence) {
    	
    	double[] probs = sentenceTokenProbs(sentence);
    	int sentenceLength = sentence.size();
    	assert(probs.length == sentenceLength + 2);
    	
    	double sentenceLogProb = 0;
    	for(int i = 1; i <= sentenceLength; ++i) {
    		double logProb = Math.log(probs[i]);
    		sentenceLogProb += logProb;
    	}
    	double startProb = probs[0];
    	double endProb = probs[sentenceLength + 1];
    	sentenceLogProb += Math.log( ( startProb + endProb ) / 2);
    	return sentenceLogProb;
    }
    
    /**
     * Compute log probability of sentence given current model
     * Similar to sentenceLogProb but exclude boundary prediction
     */
    @Override
    public double sentenceLogProb2 (List<String> sentence) {
    	
    	double[] probs = sentenceTokenProbs(sentence);
    	assert(probs.length == sentence.size() + 2);
    	
    	double sentenceLogProb = 0;
    	for(int i = 1; i <= sentence.size(); ++i) {
    		double logProb = Math.log(probs[i]);
    		sentenceLogProb += logProb;
    	}
    	return sentenceLogProb;
    }
    
    /**
     * Compute probability of every token
     * Including <S> and </S>
     */
    @Override
    public double[] sentenceTokenProbs (List<String> sentence) {
    	double[] forwardProbs = bigramModel.sentenceTokenProbs(sentence);
        double[] backwardProbs = backwardBigramModel.sentenceTokenProbs(sentence);
        int len = forwardProbs.length;
        assert(len == backwardProbs.length && len == sentence.size() + 1);
        
        // forwardProbs consists of bigram [(<S>, w_0), (w_0, w_1) ... (w_size-1, <\S>)];
        //                       and unigram [w_0, w_1 ... w_size-1, <\S>]
        // backwardProbs consists of [(<\S>, w_size-1), (w_size-1, w_size-2) ... (<w_0>, <S>)];
        //                       and unigram [w_size-1, w_size-2 ... w_0 <S>]
        // here, size is sentecne.size(). and we assert len == size + 1
        // So forwardProbs[i] is corresponding to backwardProbs[len - i - 2]
        double[] probs = new double[len + 1];
  
        probs[0] = backwardProbs[len - 1]; //Probability of start
        probs[len] = forwardProbs[len - 1]; //Probability of end
        for(int i = 1; i < len; ++i) {
        	//Probability ot w_i
        	probs[i] = forwardProbs[i-1] * forwardWeight + backwardProbs[len - i - 1] * backwardWeight;
        }
        return probs;
    }
    
	/** Train and test a bigram model.
     *  Command format: "nlp.lm.BidirectionalBigramModel [DIR]* [TestFrac]" where DIR 
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
		BidirectionalBigramModel model = new BidirectionalBigramModel();
		System.out.println("Training...");
		model.train(trainSentences);
		// Test on training data using test and test2
		//model.test(trainSentences);
		model.test2(trainSentences);
		System.out.println("Testing...");
		// Test on test data using test and test2
		//model.test(testSentences);
		model.test2(testSentences);
		
    }
}
