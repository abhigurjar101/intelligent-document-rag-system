import tempfile
import unittest
from pathlib import Path
from rag_engine import DocumentRAGEngine, chunk_document

class RetrievalTests(unittest.TestCase):
    def test_second_document_preserves_first_vocabulary(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = DocumentRAGEngine()
            for name, text in [("one.txt", "Orchids require humidity."),
                               ("two.txt", "Satellites orbit planets.")]:
                path = Path(directory) / name
                path.write_text(text)
                engine.ingest_file(str(path))
            result = engine.query("orchids")
            self.assertEqual(result["citations"], ["one.txt#clause_1"])
            self.assertIn("Orchids", result["answer"])
            self.assertEqual(engine.query("volcanoes")["citations"], [])

    def test_chunk_size_and_empty_document(self):
        self.assertEqual(chunk_document(""), [])
        self.assertTrue(all(len(c.split()) <= 3 for c in chunk_document("one two three four five six seven", 3)))
        with self.assertRaises(ValueError):
            chunk_document("text", 0)

    def test_invalid_query_arguments(self):
        engine = DocumentRAGEngine()
        for k in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError):
                engine.query("test", k)
        with self.assertRaises(ValueError):
            engine.query(" ")
