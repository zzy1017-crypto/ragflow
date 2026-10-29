//go:build integration && e2e

package api

import (
	"context"
	"math"
	"net/http"
	"slices"
	"strings"
	"testing"
	"time"
)

// Requires the live Go API, ingestor, storage, search engine and embedding service.
// Both tags are needed because the shared REST test helpers use integration.
func TestRESTRetrievalIngestionE2E(t *testing.T) {
	paragraphs := []string{
		"Coriander botanical archive entry alpha records fresh green leaves.",
		"Coriander botanical archive entry beta describes dried seeds used in cooking.",
		"Coriander botanical archive entry gamma preserves planting dates and harvest notes.",
	}
	dataset := createRESTDataset(t, apiObject{
		"embedding_model": defaultEmbedModel + "@Builtin",
		"parser_config": apiObject{
			"GeneralChunker:SixApplesFall": apiObject{
				// A custom delimiter keeps each paragraph as one chunk without merging.
				"delimiters":         []string{"`\n\n`"},
				"overlapped_percent": 0,
				"parent_child":       apiObject{"use_parent_child": false},
			},
		},
	})
	datasetID := dataset["id"].(string)
	file := documentUpload{Name: "retrieval_ingestion.txt", Content: []byte(strings.Join(paragraphs, "\n\n"))}
	document := object(t, uploadRESTDocuments(t, TestConfig, datasetID, []documentUpload{file})[0])
	documentID := document["id"].(string)
	path := documentRESTPath(datasetID)
	parsed := object(t, restOK(t, restCall(t, TestConfig, http.MethodPost, path+"/parse", apiObject{"document_ids": []string{documentID}}, nil))["data"])
	if parsed["success_count"] != float64(1) {
		t.Fatalf("accepted document count=%v want=1", parsed)
	}
	completed := waitRESTDocumentStatus(t, datasetID, []string{documentID}, "COMPLETED")[0]
	if completed["chunk_count"] != float64(len(paragraphs)) || completed["token_count"].(float64) <= 0 {
		t.Fatalf("completed document lacks the three expected chunks or tokens: %v", completed)
	}

	body := apiObject{
		"dataset_ids": []string{datasetID}, "question": "coriander botanical archive",
		"page_size": len(paragraphs), "similarity_threshold": 0,
		"vector_similarity_weight": 1, "highlight": true, "include_knowledge_compilation": false,
	}
	// Completion and search visibility can differ; retry only missing index hits.
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	ticker := time.NewTicker(300 * time.Millisecond)
	defer ticker.Stop()
	var data apiObject
	var chunks []apiObject
	for {
		data, chunks = retrievalChunks(t, restCallContext(t, ctx, TestConfig, http.MethodPost, "/retrieval", body, nil))
		if len(chunks) >= len(paragraphs) {
			break
		}
		select {
		case <-ctx.Done():
			t.Fatalf("index visibility timeout: got %d chunks want %d; data=%v", len(chunks), len(paragraphs), data)
		case <-ticker.C:
		}
	}

	assertStats := func(t *testing.T, data apiObject) {
		t.Helper()
		if data["total"] != float64(len(paragraphs)) {
			t.Fatalf("total=%v want=%d", data["total"], len(paragraphs))
		}
		aggs := array(t, data["doc_aggs"])
		if len(aggs) != 1 {
			t.Fatalf("doc_aggs=%v want one document", aggs)
		}
		agg := object(t, aggs[0])
		if agg["doc_id"] != documentID || agg["doc_name"] != file.Name || agg["count"] != float64(len(paragraphs)) {
			t.Fatalf("incorrect document aggregation: %v", agg)
		}
	}
	assertStats(t, data)
	f := retrievalFixture{datasetID: datasetID, documentIDs: []string{documentID}, content: map[string]string{}}
	var contents []string
	for _, chunk := range chunks {
		id, content := chunk["id"].(string), chunk["content"].(string)
		f.chunkIDs = append(f.chunkIDs, id)
		f.content[id] = content
		contents = append(contents, content)
		// A lexical hit alone must not hide missing document/query embeddings.
		score, ok := chunk["vector_similarity"].(float64)
		if !ok || math.IsNaN(score) || math.IsInf(score, 0) || score <= 0 {
			t.Fatalf("chunk %s has no valid positive vector similarity: %v", id, chunk["vector_similarity"])
		}
	}
	slices.Sort(contents)
	wantContents := slices.Clone(paragraphs)
	slices.Sort(wantContents)
	if !slices.Equal(contents, wantContents) {
		t.Fatalf("retrieved contents=%q want=%q", contents, wantContents)
	}
	assertRESTRetrievalFixture(t, f, chunks)

	t.Run("highlight", func(t *testing.T) {
		for _, chunk := range chunks {
			highlight, ok := chunk["highlight"].(string)
			if !ok || !strings.Contains(strings.ToLower(highlight), "<em>coriander</em>") {
				t.Fatalf("missing useful highlight: %v", chunk["highlight"])
			}
		}
	})
	t.Run("pagination", func(t *testing.T) {
		body["page_size"] = 2
		var paged []apiObject
		for page, wantCount := range []int{2, 1} {
			body["page"] = page + 1
			data, pageChunks := retrievalChunks(t, restCall(t, TestConfig, http.MethodPost, "/retrieval", body, nil))
			if len(pageChunks) != wantCount {
				t.Fatalf("page %d has %d chunks want %d", page+1, len(pageChunks), wantCount)
			}
			assertStats(t, data)
			paged = append(paged, pageChunks...)
		}
		// Exact IDs and contents also catch repeated or missing chunks across pages.
		assertRESTRetrievalFixture(t, f, paged)
	})
}
