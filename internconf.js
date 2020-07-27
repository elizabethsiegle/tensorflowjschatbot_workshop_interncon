const tf = require("@tensorflow/tfjs");
const fetch = require("node-fetch");
exports.handler = async function(context, event, callback) {
	let twiml = new Twilio.twiml.MessagingResponse();
	const model = await loadModel(); 
	const metadata = await getMetaData();
	let sum = 0;
	const prediction = event.Body.toLowerCase().trim();
	console.log(`${prediction}`);
    let perc = predict(prediction, model, metadata);
    sum += parseFloat(perc, 10);
    console.log(`sum ${sum}`);
	
	console.log(getSentiment(sum));
	twiml.message(getSentiment(sum));
	callback(null, twiml);
};
const getMetaData = async () => {
  const metadata = await fetch("https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json")
  return metadata.json()
}
const padSequences = (sequences, metadata) => {
  return sequences.map(seq => {
    if (seq.length > metadata.max_len) {
      seq.splice(0, seq.length - metadata.max_len);
    }
    if (seq.length < metadata.max_len) {
      const pad = [];
      for (let i = 0; i < metadata.max_len - seq.length; ++i) {
        pad.push(0);
      }
      seq = pad.concat(seq);
    }
    return seq;
  });
}
const loadModel = async () => {
    const url = `https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json`;
    const model = await tf.loadLayersModel(url);
    return model;
};
const predict = (text, model, metadata) => {
  const trimmed = text.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split(' ');
  const sequence = trimmed.map(word => {
    const wordIndex = metadata.word_index[word];
    if (typeof wordIndex === 'undefined') {
      return 2; //oov_index
    }
    return wordIndex + metadata.index_from;
  });
  const paddedSequence = padSequences([sequence], metadata);
  const input = tf.tensor2d(paddedSequence, [1, metadata.max_len]);

  const predictOut = model.predict(input);
  const score = predictOut.dataSync()[0];
  predictOut.dispose();
  return score;
}
const songArrayPos = [
    "I was so ahead of the curve, the curve became a sphere",
    "I’d give you my sunshine, give you my best, but the rain is always gonna come if you’re standing with me.",
    "I’ve never been a natural, all I do is try, try, try.",
    "I had a marvelous time ruining everything.",
    "Time, wondrous time, gave me the blues and then purple-pink skies."
    ];
const songArrayNeutral = [
    "Our coming of age has come and gone.",
    "They told me all of my cages were mental, so I got wasted like all my potential.",
    "You’re not my homeland anymore, so what am I defending now?",
    "When I felt like I was an old cardigan under someone’s bed, you put me on and said I was your favorite.",
    "It would’ve been fun if you would’ve been the one."
    ];
const songArrayNeg = [
    "We gather stones never knowing what they’ll mean. Some to throw, some to make a diamond ring.",
    "If I’m dead to you why are you at the wake?",
    "And women like hunting witches too, doing your dirtiest work for you.",
    "And if I’m on fire, you’ll be made to ashes too.",
    "Hell was the journey but it brought me heaven."
    ];
const getSentiment = (score) => {
  if (score > 0.66) {
    return `Score of ${score} is Positive. Here's a selected Taylor Swift folklore lyric: ${songArrayPos[songArrayPos.length * Math.random() | 0]}`;
  }
  else if (score > 0.4) {
    return `Score of ${score} is Neutral. Here's a selected Taylor Swift folklore lyric: ${songArrayNeutral[songArrayNeutral.length * Math.random() | 0]}`;
  }
  else {
    return `Score of ${score} is Negative. Here's a selected Taylor Swift folklore lyric: ${songArrayNeg[songArrayNeg.length * Math.random() | 0]}`;
  }
}