import asyncio
import timeit
import concurrent.futures
import logging
from typing import List
import readline

import tensorflow as tf
import tensorflow_hub as hub
import sentencepiece as spm

logger = logging.getLogger(__name__)
GPU_USAGE = .4


class UniversalSentenceEncoderLite():
    """Encodes a batch of sentences using the Universal Sentence Encoder model
    as described in https://arxiv.org/abs/1803.11175
    """

    # TODO: https://stackoverflow.com/questions/33128325/how-to-set-class-attribute-with-await-in-init ?
    def __init__(
            self,
            url="https://tfhub.dev/google/universal-sentence-encoder-lite/1"):
        self.url = url
        self._graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = GPU_USAGE
        with self._graph.as_default():
            self._session = tf.Session(config=config)
            self._sp = spm.SentencePieceProcessor()
            self._embed = None
            self._embeddings = None

    def __enter__(self):
        self._embed = self._load_models()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self._session.close()
        tf.reset_default_graph()

    def _convert_to_ids(self, sentences):
        """Utility method that processes sentences with the sentence piece
        processor.
        Args:
            sentences (:obj:`list` of :obj:`str`): The sentences to process.
        Returns: the processed results in tf.SparseTensor-similar format:
            (values, indices, dense_shape).
        """
        ids = [self._sp.EncodeAsIds(x) for x in sentences]
        max_len = max(len(x) for x in ids)
        dense_shape = (len(ids), max_len)
        values = [item for sublist in ids for item in sublist]
        indices = [[row, col] for row in range(len(ids))
                   for col in range(len(ids[row]))]
        return (values, indices, dense_shape)

    async def load_models(self, loop):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            tasks = [loop.run_in_executor(executor, self._load_models)]
            completed, _ = await asyncio.wait(tasks)
            results = [t.result() for t in completed]
            self._embed = results[0]
            logger.debug("set the model function")

    def _load_models(self):
        logger.info("loading models...")
        start = timeit.default_timer()
        with self._graph.as_default():
            _embed = hub.Module(self.url)

            self._sp.Load(self._session.run(_embed(signature="spm_path")))

            self._input_placeholder = tf.sparse_placeholder(
                tf.int64, shape=[None, None])

            self._embeddings = _embed(
                inputs={
                    "values": self._input_placeholder.values,
                    "indices": self._input_placeholder.indices,
                    "dense_shape": self._input_placeholder.dense_shape
                })

            self._session.run([
                tf.global_variables_initializer(),
                tf.tables_initializer(),
            ])

            logger.info("loaded model in %.4fs",
                        timeit.default_timer() - start)
            return _embed

    def embed(self, sentences: List[str]):
        """Run the tensorflow graph to embed the sentences.
        Args:
            sentences (:obj:`list` of :obj:`str`): The sentences to embed.
        Returns:
            :obj:`list` of :obj:`list` of :obj:`float`: The embeddings
                corresponding to each sentence.
        """
        with self._graph.as_default():
            if not self._embed:
                logger.warning("attempted embed() when model was not loaded")
                self._embed = self._load_models()

            values, indices, dense_shape = self._convert_to_ids(sentences)

            return self._session.run(
                self._embeddings,
                feed_dict={
                    self._input_placeholder.values: values,
                    self._input_placeholder.indices: indices,
                    self._input_placeholder.dense_shape: dense_shape
                })


class UniversalSentenceEncoderLarge():
    """Encodes a batch of sentences using the Universal Sentence Encoder model
    as described in https://arxiv.org/abs/1803.11175
    """

    # TODO: https://stackoverflow.com/questions/33128325/how-to-set-class-attribute-with-await-in-init ?
    def __init__(
            self,
            url="https://tfhub.dev/google/universal-sentence-encoder-large/1"):
        self.url = url
        self._graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = GPU_USAGE
        with self._graph.as_default():
            self._session = tf.Session(config=config)
        self._embed = None

    # TODO: refactor - DRY with embed wrapper
    async def load_models(self, loop):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            tasks = [loop.run_in_executor(executor, self._load_models)]
            completed, _ = await asyncio.wait(tasks)
            results = [t.result() for t in completed]
            self._embed = results[0]
            logger.debug("set the model function")

    def _load_models(self):
        logger.info("loading models...")
        start = timeit.default_timer()
        with self._graph.as_default():
            _embed = hub.Module(self.url)
            self._session.run([
                tf.global_variables_initializer(),
                tf.tables_initializer(),
            ])
            logger.info("loaded model in %.4fs",
                        timeit.default_timer() - start)
            return _embed

    def __enter__(self):
        self._embed = self._load_models()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def embed(self, sentences: List[str]):
        """Run the tensorflow graph to embed the sentences.
        Args:
            sentences (:obj:`list` of :obj:`str`): The sentences to embed.
        Returns:
            :obj:`list` of :obj:`list` of :obj:`float`: The embeddings corresponding to each sentence.
        """
        with self._graph.as_default():
            if not self._embed:
                logger.warning("attempted embed() when model was not loaded")
                self._embed = self._load_models()
            return self._session.run(self._embed(sentences))

    def close(self):
        self._session.close()
        tf.reset_default_graph()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    with UniversalSentenceEncoderLarge() as encoder_large:
        with UniversalSentenceEncoderLite() as encoder_lite:
            while True:
                sentence = [input("> ")]
                start = timeit.default_timer()
                # embeddings = encoder.embed([
                #     "The quick brown fox jumps over the lazy dog.",
                #     "I am a sentence for which I would like to get its embedding",
                #     "I am a sentence.",
                #     "I am another sentence.",
                #     "I am a camera.",
                #     "I love lamp.",
                #     "Don't kill the whale."
                # ])  # yapf: disable
                embeddings_lite = encoder_lite.embed(sentence)
                print(embeddings_lite, "LITE", timeit.default_timer() - start, sep="\n")
                embeddings_large = encoder_large.embed(sentence)
                print(embeddings_large, "LARGE", timeit.default_timer() - start, sep="\n")