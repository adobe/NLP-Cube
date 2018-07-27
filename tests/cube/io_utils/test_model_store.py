import os
import unittest2
import httpretty

# Append parent dir to sys path.
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(1, parentdir)

from cube.io_utils.model_store import ModelStore


XML_CONTENT_FOR_GET_LATEST_VERSION = (
    b'\xef\xbb\xbf<?xml version="2.0" encoding="utf-8"?><EnumerationResults ServiceEndpoint="'
    b'https://nlpcube.blob.core.windows.net/" ContainerName="models"><Blobs><Blob><Name>en-2.0.zip'
    b'</Name><Properties><Last-Modified>Fri, 27 Jul 2018 11:45:57 GMT</Last-Modified>'
    b'<Etag>0x8D5F3B684C5345A</Etag><Content-Length>310078470</Content-Length>'
    b'<Content-Type>application/zip</Content-Type><Content-Encoding /><Content-Language />'
    b'<Content-MD5>zrLJ7dMMbF6wYO9TaZr8GQ==</Content-MD5><Cache-Control /><Content-Disposition />'
    b'<BlobType>BlockBlob</BlobType><LeaseStatus>unlocked</LeaseStatus><LeaseState>available</LeaseState>'
    b'</Properties></Blob><Blob><Name>ro-1.0.zip</Name><Properties>'
    b'<Last-Modified>Tue, 24 Jul 2018 15:40:34 GMT</Last-Modified><Etag>0x8D5F17BCBAFCD25</Etag>'
    b'<Content-Length>310078470</Content-Length><Content-Type>application/zip</Content-Type>'
    b'<Content-Encoding /><Content-Language /><Content-MD5>zrLJ7dMMbF6wYO9TaZr8GQ==</Content-MD5>'
    b'<Cache-Control /><Content-Disposition /><BlobType>BlockBlob</BlobType>'
    b'<LeaseStatus>unlocked</LeaseStatus><LeaseState>available</LeaseState></Properties></Blob></Blobs>'
    b'<NextMarker /></EnumerationResults>'
)


class BackendTestCase(unittest2.TestCase):

    @httpretty.activate
    def test_get_latest_model_versions(self):
        httpretty.register_uri(
            httpretty.GET,
            ModelStore.MODELS_PATH_CLOUD_ALL,
            body=XML_CONTENT_FOR_GET_LATEST_VERSION
        )

        model_store = ModelStore()
        latest_model_versions = model_store.get_latest_model_versions()
        self.assertEqual(latest_model_versions, {'en': '2.0', 'ro': '1.0'})
