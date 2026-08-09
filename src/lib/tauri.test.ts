// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { invoke } from '@tauri-apps/api/core';
import * as tauri from './tauri';
import { daemonMeetsFloor } from "./tauri";

const mockInvoke = vi.mocked(invoke);

beforeEach(() => {
  mockInvoke.mockReset();
  mockInvoke.mockResolvedValue(undefined as any);
});

describe('search', () => {
  it('calls invoke with explicit args', async () => {
    mockInvoke.mockResolvedValue([]);
    await tauri.search('test query', 5, 'local_files');
    expect(mockInvoke).toHaveBeenCalledWith('search', {
      query: 'test query',
      limit: 5,
      sourceFilter: 'local_files',
    });
  });

  it('defaults limit to 10 and sourceFilter to null when not provided', async () => {
    mockInvoke.mockResolvedValue([]);
    await tauri.search('hello');
    expect(mockInvoke).toHaveBeenCalledWith('search', {
      query: 'hello',
      limit: 10,
      sourceFilter: null,
    });
  });
});

describe('getIndexStatus', () => {
  it('calls invoke with no args', async () => {
    await tauri.getIndexStatus();
    expect(mockInvoke).toHaveBeenCalledWith('get_index_status');
  });
});

describe('getPipelineStatus', () => {
  it('calls invoke with no args', async () => {
    mockInvoke.mockResolvedValue({
      enrichment: {},
      entity_linking: { linked: 0, unlinked: 0 },
      refinement_queue: [],
      recaps: 0,
      types: {},
      quality: {},
    });

    await tauri.getPipelineStatus();

    expect(mockInvoke).toHaveBeenCalledWith('get_pipeline_status');
  });
});

describe('addWatchPath', () => {
  it('calls invoke with path arg', async () => {
    await tauri.addWatchPath('/home/user/docs');
    expect(mockInvoke).toHaveBeenCalledWith('add_watch_path', { path: '/home/user/docs' });
  });
});

describe('removeWatchPath', () => {
  it('calls invoke with path arg', async () => {
    await tauri.removeWatchPath('/home/user/docs');
    expect(mockInvoke).toHaveBeenCalledWith('remove_watch_path', { path: '/home/user/docs' });
  });
});

describe('listWatchPaths', () => {
  it('calls invoke with no args', async () => {
    mockInvoke.mockResolvedValue([]);
    await tauri.listWatchPaths();
    expect(mockInvoke).toHaveBeenCalledWith('list_watch_paths');
  });
});

describe('registered sources', () => {
  it('listRegisteredSources calls invoke with no args', async () => {
    mockInvoke.mockResolvedValue([]);
    await tauri.listRegisteredSources();
    expect(mockInvoke).toHaveBeenCalledWith('list_registered_sources');
  });

  it('addSource preserves sourceType arg casing', async () => {
    const source = {
      id: 'obsidian-vault',
      source_type: 'obsidian',
      path: '/Users/test/vault',
      status: 'Active',
      last_sync: null,
      file_count: 0,
      memory_count: 0,
    };
    mockInvoke.mockResolvedValue(source);

    await expect(tauri.addSource('obsidian', '/Users/test/vault')).resolves.toEqual(source);

    expect(mockInvoke).toHaveBeenCalledWith('add_source', {
      sourceType: 'obsidian',
      path: '/Users/test/vault',
    });
  });

  it('removeSource passes id', async () => {
    await tauri.removeSource('obsidian-vault');
    expect(mockInvoke).toHaveBeenCalledWith('remove_source', { id: 'obsidian-vault' });
  });

  it('syncRegisteredSource passes id', async () => {
    const stats = { files_found: 4, ingested: 2, skipped: 1, errors: 0 };
    mockInvoke.mockResolvedValue(stats);

    await expect(tauri.syncRegisteredSource('obsidian-vault')).resolves.toEqual(stats);

    expect(mockInvoke).toHaveBeenCalledWith('sync_registered_source', {
      id: 'obsidian-vault',
    });
  });
});

describe('deleteFileChunks', () => {
  it('calls invoke with source and sourceId', async () => {
    await tauri.deleteFileChunks('local_files', 'abc123');
    expect(mockInvoke).toHaveBeenCalledWith('delete_file_chunks', {
      source: 'local_files',
      sourceId: 'abc123',
    });
  });
});

describe('deleteByTimeRange', () => {
  it('calls invoke with start and end timestamps', async () => {
    await tauri.deleteByTimeRange(1000000, 2000000);
    expect(mockInvoke).toHaveBeenCalledWith('delete_by_time_range', {
      start: 1000000,
      end: 2000000,
    });
  });
});

describe('quickCapture', () => {
  it('calls invoke with req object', async () => {
    mockInvoke.mockResolvedValue(42);
    const req = { title: 'My note', content: 'Some content', tags: ['tag1'] };
    const result = await tauri.quickCapture(req);
    expect(mockInvoke).toHaveBeenCalledWith('quick_capture', { req });
    expect(result).toBe(42);
  });

  it('works with minimal req (content only)', async () => {
    mockInvoke.mockResolvedValue(1);
    await tauri.quickCapture({ content: 'Just content' });
    expect(mockInvoke).toHaveBeenCalledWith('quick_capture', {
      req: { content: 'Just content' },
    });
  });
});

describe('listAllTags', () => {
  it('calls invoke with no args', async () => {
    await tauri.listAllTags();
    expect(mockInvoke).toHaveBeenCalledWith('list_all_tags');
  });
});

describe('setDocumentTags', () => {
  it('calls invoke with source, sourceId, and tags', async () => {
    mockInvoke.mockResolvedValue(['tag1', 'tag2']);
    await tauri.setDocumentTags('local_files', 'doc1', ['tag1', 'tag2']);
    expect(mockInvoke).toHaveBeenCalledWith('set_document_tags', {
      source: 'local_files',
      sourceId: 'doc1',
      tags: ['tag1', 'tag2'],
    });
  });
});

describe('createEntity', () => {
  it('calls invoke with name, entityType, and explicit domain', async () => {
    mockInvoke.mockResolvedValue('entity-id-1');
    await tauri.createEntity('Alice', 'person', 'work');
    expect(mockInvoke).toHaveBeenCalledWith('create_entity_cmd', {
      name: 'Alice',
      entityType: 'person',
      domain: 'work',
    });
  });

  it('defaults domain to null when not provided', async () => {
    mockInvoke.mockResolvedValue('entity-id-2');
    await tauri.createEntity('Bob', 'person');
    expect(mockInvoke).toHaveBeenCalledWith('create_entity_cmd', {
      name: 'Bob',
      entityType: 'person',
      domain: null,
    });
  });
});

describe('addObservation', () => {
  it('calls invoke with all explicit args', async () => {
    mockInvoke.mockResolvedValue('obs-id-1');
    await tauri.addObservation('entity-1', 'Likes coffee', 'claude', 0.9);
    expect(mockInvoke).toHaveBeenCalledWith('add_observation_cmd', {
      entityId: 'entity-1',
      content: 'Likes coffee',
      sourceAgent: 'claude',
      confidence: 0.9,
    });
  });

  it('defaults sourceAgent and confidence to null when not provided', async () => {
    mockInvoke.mockResolvedValue('obs-id-2');
    await tauri.addObservation('entity-1', 'Likes tea');
    expect(mockInvoke).toHaveBeenCalledWith('add_observation_cmd', {
      entityId: 'entity-1',
      content: 'Likes tea',
      sourceAgent: null,
      confidence: null,
    });
  });
});

describe('listMemoriesRich', () => {
  it('calls invoke with all explicit args', async () => {
    mockInvoke.mockResolvedValue([]);
    await tauri.listMemoriesRich('work', 'fact', true, 50);
    expect(mockInvoke).toHaveBeenCalledWith('list_memories_cmd', {
      domain: 'work',
      memoryType: 'fact',
      confirmed: true,
      limit: 50,
    });
  });

  it('defaults all optional args to null when not provided', async () => {
    mockInvoke.mockResolvedValue([]);
    await tauri.listMemoriesRich();
    expect(mockInvoke).toHaveBeenCalledWith('list_memories_cmd', {
      domain: null,
      memoryType: null,
      confirmed: null,
      limit: null,
    });
  });
});

describe('updateMemory', () => {
  it('calls invoke with all explicit args', async () => {
    await tauri.updateMemory('src-id-1', 'New content', 'personal', true, 'fact');
    expect(mockInvoke).toHaveBeenCalledWith('update_memory_cmd', {
      sourceId: 'src-id-1',
      content: 'New content',
      domain: 'personal',
      confirmed: true,
      memoryType: 'fact',
    });
  });

  it('defaults optional args to null when not provided', async () => {
    await tauri.updateMemory('src-id-2');
    expect(mockInvoke).toHaveBeenCalledWith('update_memory_cmd', {
      sourceId: 'src-id-2',
      content: null,
      domain: null,
      confirmed: null,
      memoryType: null,
    });
  });
});

describe('getProfile', () => {
  it('calls invoke with no args', async () => {
    mockInvoke.mockResolvedValue(null);
    await tauri.getProfile();
    expect(mockInvoke).toHaveBeenCalledWith('get_profile');
  });
});

describe('listAgents', () => {
  it('calls invoke with no args', async () => {
    mockInvoke.mockResolvedValue([]);
    await tauri.listAgents();
    expect(mockInvoke).toHaveBeenCalledWith('list_agents');
  });
});

describe('deleteAgent', () => {
  it('calls invoke with name arg', async () => {
    await tauri.deleteAgent('my-agent');
    expect(mockInvoke).toHaveBeenCalledWith('delete_agent', { name: 'my-agent' });
  });
});

describe('pinMemory', () => {
  it('calls invoke with sourceId', async () => {
    await tauri.pinMemory('src-id-pin');
    expect(mockInvoke).toHaveBeenCalledWith('pin_memory', { sourceId: 'src-id-pin' });
  });
});

describe('unpinMemory', () => {
  it('calls invoke with sourceId', async () => {
    await tauri.unpinMemory('src-id-unpin');
    expect(mockInvoke).toHaveBeenCalledWith('unpin_memory', { sourceId: 'src-id-unpin' });
  });
});

describe('enrichment status', () => {
  it('getEnrichmentStatus passes sourceId', async () => {
    mockInvoke.mockResolvedValue({
      source_id: 'mem-1',
      summary: 'complete',
      steps: [{ step: 'classify', status: 'done', error: null, attempts: 1 }],
    });

    const result = await tauri.getEnrichmentStatus('mem-1');

    expect(mockInvoke).toHaveBeenCalledWith('get_enrichment_status', {
      sourceId: 'mem-1',
    });
    expect(result.summary).toBe('complete');
  });
});

describe('refinery queue', () => {
  it('lists daemon refinery proposals with a limit', async () => {
    const response = {
      proposals: [
        {
          id: 'ref-1',
          action: 'entity_merge',
          source_ids: ['mem-a', 'mem-b'],
          payload: {
            action: 'entity_merge',
            existing_id: 'ent-a',
            new_id: 'ent-b',
            similarity: 0.86,
          },
          confidence: 0.86,
          created_at: '2026-06-26T00:00:00Z',
        },
      ],
    };
    mockInvoke.mockResolvedValue(response);

    await expect(tauri.listRefinements(6)).resolves.toEqual(response);

    expect(mockInvoke).toHaveBeenCalledWith('list_refinements', { limit: 6 });
  });

  it('accepts and rejects daemon refinery proposals by id', async () => {
    mockInvoke
      .mockResolvedValueOnce({ id: 'ref-1', action_applied: 'entity_merge' })
      .mockResolvedValueOnce({ id: 'ref-1' });

    await expect(tauri.acceptRefinement('ref-1')).resolves.toEqual({
      id: 'ref-1',
      action_applied: 'entity_merge',
    });
    await expect(tauri.rejectRefinement('ref-1')).resolves.toEqual({ id: 'ref-1' });

    expect(mockInvoke).toHaveBeenNthCalledWith(1, 'accept_refinement', { id: 'ref-1' });
    expect(mockInvoke).toHaveBeenNthCalledWith(2, 'reject_refinement', { id: 'ref-1' });
  });

  it("keeps legacy entity suggestion accept unsupported until the daemon has an accept path", async () => {
    mockInvoke.mockResolvedValueOnce({ id: "ref-suggest" });

    await expect(tauri.approveEntitySuggestion("ref-suggest")).rejects.toThrow(
      "Entity suggestion accept is not supported by this daemon contract",
    );
    await expect(tauri.dismissEntitySuggestion("ref-suggest")).resolves.toEqual({ id: "ref-suggest" });

    expect(mockInvoke).toHaveBeenCalledTimes(1);
    expect(mockInvoke).toHaveBeenNthCalledWith(1, "reject_refinement", { id: "ref-suggest" });
  });
});

// --- Additional wrappers for coverage ---

describe('reindex', () => {
  it('calls invoke', async () => {
    await tauri.reindex();
    expect(mockInvoke).toHaveBeenCalledWith('reindex');
  });
});

describe('listIndexedFiles', () => {
  it('calls invoke', async () => {
    mockInvoke.mockResolvedValue([]);
    await tauri.listIndexedFiles();
    expect(mockInvoke).toHaveBeenCalledWith('list_indexed_files');
  });
});

describe('deleteBulk', () => {
  it('maps sourceId to source_id', async () => {
    await tauri.deleteBulk([{ source: 's1', sourceId: 'id1' }]);
    expect(mockInvoke).toHaveBeenCalledWith('delete_bulk', {
      items: [{ source: 's1', source_id: 'id1' }],
    });
  });
});

describe('openFile', () => {
  it('strips file:// prefix', async () => {
    await tauri.openFile('file:///tmp/test.txt');
    expect(mockInvoke).toHaveBeenCalledWith('open_file', { path: '/tmp/test.txt' });
  });

  it('passes non-prefixed paths directly', async () => {
    await tauri.openFile('/tmp/test.txt');
    expect(mockInvoke).toHaveBeenCalledWith('open_file', { path: '/tmp/test.txt' });
  });
});

describe('getChunks', () => {
  it('calls invoke with source and sourceId', async () => {
    mockInvoke.mockResolvedValue([]);
    await tauri.getChunks('local_files', 'doc1');
    expect(mockInvoke).toHaveBeenCalledWith('get_chunks', { source: 'local_files', sourceId: 'doc1' });
  });
});

describe('updateChunk', () => {
  it('calls invoke with id and content', async () => {
    await tauri.updateChunk('chunk-1', 'updated content');
    expect(mockInvoke).toHaveBeenCalledWith('update_chunk', { id: 'chunk-1', content: 'updated content' });
  });
});

describe('activities', () => {
  it('listActivities calls invoke', async () => {
    mockInvoke.mockResolvedValue([]);
    await tauri.listActivities();
    expect(mockInvoke).toHaveBeenCalledWith('list_activities');
  });

  it('rebuildActivities calls invoke', async () => {
    mockInvoke.mockResolvedValue(5);
    await tauri.rebuildActivities();
    expect(mockInvoke).toHaveBeenCalledWith('rebuild_activities');
  });
});

describe('getCaptureStats', () => {
  it('calls invoke', async () => {
    mockInvoke.mockResolvedValue({});
    await tauri.getCaptureStats();
    expect(mockInvoke).toHaveBeenCalledWith('get_capture_stats');
  });
});

describe('deleteTag', () => {
  it('passes name', async () => {
    await tauri.deleteTag('old-tag');
    expect(mockInvoke).toHaveBeenCalledWith('delete_tag', { name: 'old-tag' });
  });
});

describe('suggestTags', () => {
  it('passes all args', async () => {
    mockInvoke.mockResolvedValue(['suggested']);
    await tauri.suggestTags('local_files', 'doc1', 12345);
    expect(mockInvoke).toHaveBeenCalledWith('suggest_tags', { source: 'local_files', sourceId: 'doc1', lastModified: 12345 });
  });
});

describe('spaces', () => {
  it('listSpaces calls invoke', async () => {
    mockInvoke.mockResolvedValue([]);
    await tauri.listSpaces();
    expect(mockInvoke).toHaveBeenCalledWith('list_spaces');
  });

  it('getSpace passes name', async () => {
    mockInvoke.mockResolvedValue(null);
    await tauri.getSpace('work');
    expect(mockInvoke).toHaveBeenCalledWith('get_space', { name: 'work' });
  });

  it('createSpace passes name and description', async () => {
    mockInvoke.mockResolvedValue({ id: '1', name: 'work', description: 'Work stuff', suggested: false, memory_count: 0, entity_count: 0, created_at: 0, updated_at: 0 });
    await tauri.createSpace('work', 'Work stuff');
    expect(mockInvoke).toHaveBeenCalledWith('create_space', { name: 'work', description: 'Work stuff' });
  });

  it('updateSpace passes name, newName and description', async () => {
    mockInvoke.mockResolvedValue({ id: '1', name: 'career', description: 'Career stuff', suggested: false, memory_count: 0, entity_count: 0, created_at: 0, updated_at: 0 });
    await tauri.updateSpace('work', 'career', 'Career stuff');
    expect(mockInvoke).toHaveBeenCalledWith('update_space', { name: 'work', newName: 'career', description: 'Career stuff' });
  });

  it('deleteSpace passes only the space name supported by the daemon API', async () => {
    await tauri.deleteSpace('work');
    expect(mockInvoke).toHaveBeenCalledWith('delete_space', { name: 'work' });
  });

  it('moveSpace passes source and destination space names', async () => {
    mockInvoke.mockResolvedValue({ affected: 7 });
    const result = await tauri.moveSpace('Inbox', 'Archive');
    expect(result).toEqual({ affected: 7 });
    expect(mockInvoke).toHaveBeenCalledWith('move_space', { from: 'Inbox', to: 'Archive' });
  });

  it('confirmSpace passes name', async () => {
    await tauri.confirmSpace('work');
    expect(mockInvoke).toHaveBeenCalledWith('confirm_space', { name: 'work' });
  });

  it('addLegacySpace passes name, icon, color', async () => {
    await tauri.addLegacySpace('Work', 'briefcase', 'blue');
    expect(mockInvoke).toHaveBeenCalledWith('add_space', { name: 'Work', icon: 'briefcase', color: 'blue' });
  });

  it('removeLegacySpace passes spaceId', async () => {
    await tauri.removeLegacySpace('space-1');
    expect(mockInvoke).toHaveBeenCalledWith('remove_space', { spaceId: 'space-1' });
  });

  it('renameLegacySpace passes spaceId and newName', async () => {
    await tauri.renameLegacySpace('space-1', 'Personal');
    expect(mockInvoke).toHaveBeenCalledWith('rename_space', { spaceId: 'space-1', newName: 'Personal' });
  });
});

describe('session snapshots', () => {
  it('getSessionSnapshots passes limit with default', async () => {
    mockInvoke.mockResolvedValue([]);
    await tauri.getSessionSnapshots();
    expect(mockInvoke).toHaveBeenCalledWith('get_session_snapshots', { limit: 10 });
  });

  it('deleteSnapshot passes snapshotId', async () => {
    await tauri.deleteSnapshot('snap-1');
    expect(mockInvoke).toHaveBeenCalledWith('delete_snapshot', { snapshotId: 'snap-1' });
  });
});

describe('reclassifyMemory', () => {
  it('passes sourceId and memoryType', async () => {
    mockInvoke.mockResolvedValue('new-source-id');
    await tauri.reclassifyMemory('src-1', 'preference');
    expect(mockInvoke).toHaveBeenCalledWith('reclassify_memory_cmd', { sourceId: 'src-1', memoryType: 'preference' });
  });
});

describe('version chain', () => {
  it('getVersionChain passes sourceId', async () => {
    mockInvoke.mockResolvedValue([]);
    await tauri.getVersionChain('src-1');
    expect(mockInvoke).toHaveBeenCalledWith('get_version_chain_cmd', { sourceId: 'src-1' });
  });
});

describe('revision history', () => {
  it('getMemoryRevisions passes sourceId', async () => {
    mockInvoke.mockResolvedValue({ current_source_id: 'mem-1', chain_depth: 1, entries: [] });
    await tauri.getMemoryRevisions('mem-1');
    expect(mockInvoke).toHaveBeenCalledWith('get_memory_revisions', { sourceId: 'mem-1' });
  });

  it('getPageRevisions passes pageId', async () => {
    mockInvoke.mockResolvedValue({
      page_id: 'page-1',
      current_version: 2,
      user_edited: false,
      stale_reason: null,
      entries: [],
    });
    await tauri.getPageRevisions('page-1');
    expect(mockInvoke).toHaveBeenCalledWith('get_page_revisions', { pageId: 'page-1' });
  });
});

describe('pending revisions', () => {
  it('listPendingRevisions passes default limit', async () => {
    mockInvoke.mockResolvedValue([]);
    await tauri.listPendingRevisions();
    expect(mockInvoke).toHaveBeenCalledWith('list_pending_revisions', { limit: null });
  });

  it('listPendingRevisions passes explicit limit', async () => {
    mockInvoke.mockResolvedValue([]);
    await tauri.listPendingRevisions(25);
    expect(mockInvoke).toHaveBeenCalledWith('list_pending_revisions', { limit: 25 });
  });

  it('acceptPendingRevision passes sourceId', async () => {
    await tauri.acceptPendingRevision('src-1');
    expect(mockInvoke).toHaveBeenCalledWith('accept_pending_revision', { sourceId: 'src-1' });
  });

  it('dismissPendingRevision passes sourceId', async () => {
    await tauri.dismissPendingRevision('src-1');
    expect(mockInvoke).toHaveBeenCalledWith('dismiss_pending_revision', { sourceId: 'src-1' });
  });

  it('getPendingRevision passes sourceId', async () => {
    mockInvoke.mockResolvedValue(null);
    await tauri.getPendingRevision('src-1');
    expect(mockInvoke).toHaveBeenCalledWith('get_pending_revision', { sourceId: 'src-1' });
  });
});

describe('entity operations', () => {
  it('listEntities passes filters with null defaults', async () => {
    mockInvoke.mockResolvedValue([]);
    await tauri.listEntities();
    expect(mockInvoke).toHaveBeenCalledWith('list_entities_cmd', { entityType: null, domain: null });
  });

  it('deleteEntity passes entityId', async () => {
    await tauri.deleteEntity('e1');
    expect(mockInvoke).toHaveBeenCalledWith('delete_entity_cmd', { entityId: 'e1' });
  });

  it('confirmEntity passes entityId and confirmed', async () => {
    await tauri.confirmEntity('e1', true);
    expect(mockInvoke).toHaveBeenCalledWith('confirm_entity_cmd', { entityId: 'e1', confirmed: true });
  });
});

describe('listPinnedMemories', () => {
  it('calls invoke', async () => {
    mockInvoke.mockResolvedValue([]);
    await tauri.listPinnedMemories();
    expect(mockInvoke).toHaveBeenCalledWith('list_pinned_memories');
  });
});

describe('setAvatar', () => {
  it('passes sourcePath', async () => {
    mockInvoke.mockResolvedValue('/path/to/avatar.png');
    await tauri.setAvatar('/tmp/photo.jpg');
    expect(mockInvoke).toHaveBeenCalledWith('set_avatar', { sourcePath: '/tmp/photo.jpg' });
  });
});

describe('updateProfile', () => {
  it('passes all args with null defaults', async () => {
    await tauri.updateProfile('p1', 'Lucian', 'Lux');
    expect(mockInvoke).toHaveBeenCalledWith('update_profile', {
      id: 'p1',
      name: 'Lucian',
      display_name: 'Lux',
      email: null,
      bio: null,
      avatar_path: null,
    });
  });
});

describe('getAgent', () => {
  it('passes name', async () => {
    mockInvoke.mockResolvedValue(null);
    await tauri.getAgent('claude');
    expect(mockInvoke).toHaveBeenCalledWith('get_agent', { name: 'claude' });
  });
});

describe('updateAgent', () => {
  it('passes name and updates with null defaults', async () => {
    await tauri.updateAgent('claude', { enabled: true, trustLevel: 'full' });
    expect(mockInvoke).toHaveBeenCalledWith('update_agent', {
      name: 'claude',
      agent_type: null,
      description: null,
      enabled: true,
      trust_level: 'full',
      display_name: null,
    });
  });
});

describe('connectSource and disconnectSource', () => {
  it('connectSource passes sourceName', async () => {
    await tauri.connectSource('local_files');
    expect(mockInvoke).toHaveBeenCalledWith('connect_source', { sourceName: 'local_files' });
  });

  it('disconnectSource passes sourceName', async () => {
    await tauri.disconnectSource('local_files');
    expect(mockInvoke).toHaveBeenCalledWith('disconnect_source', { sourceName: 'local_files' });
  });
});

describe('syncSource', () => {
  it('passes sourceName', async () => {
    await tauri.syncSource('local_files');
    expect(mockInvoke).toHaveBeenCalledWith('sync_source', { sourceName: 'local_files' });
  });
});

describe('listSources', () => {
  it('calls invoke', async () => {
    mockInvoke.mockResolvedValue([]);
    await tauri.listSources();
    expect(mockInvoke).toHaveBeenCalledWith('list_sources');
  });
});

describe('setDocumentSpace', () => {
  it('passes all args', async () => {
    await tauri.setDocumentSpace('local_files', 'doc1', 'space-1');
    expect(mockInvoke).toHaveBeenCalledWith('set_document_space', { source: 'local_files', sourceId: 'doc1', spaceId: 'space-1' });
  });
});

describe('pinLegacySpace', () => {
  it('passes spaceId', async () => {
    await tauri.pinLegacySpace('space-1');
    expect(mockInvoke).toHaveBeenCalledWith('pin_space', { spaceId: 'space-1' });
  });
});

describe('getSnapshotCaptures', () => {
  it('passes snapshotId', async () => {
    mockInvoke.mockResolvedValue([]);
    await tauri.getSnapshotCaptures('snap-1');
    expect(mockInvoke).toHaveBeenCalledWith('get_snapshot_captures', { snapshotId: 'snap-1' });
  });
});

describe('getSnapshotCapturesWithContent', () => {
  it('passes snapshotId', async () => {
    mockInvoke.mockResolvedValue([]);
    await tauri.getSnapshotCapturesWithContent('snap-1');
    expect(mockInvoke).toHaveBeenCalledWith('get_snapshot_captures_with_content', { snapshotId: 'snap-1' });
  });
});

describe('getEntityDetail', () => {
  it('passes entityId', async () => {
    mockInvoke.mockResolvedValue({ entity: {}, observations: [], relations: [] });
    await tauri.getEntityDetail('e1');
    expect(mockInvoke).toHaveBeenCalledWith('get_entity_detail_cmd', { entityId: 'e1' });
  });
});

describe('updateObservation', () => {
  it('passes observationId and content', async () => {
    await tauri.updateObservation('obs1', 'new content');
    expect(mockInvoke).toHaveBeenCalledWith('update_observation_cmd', { observationId: 'obs1', content: 'new content' });
  });
});

describe('deleteObservation', () => {
  it('passes observationId', async () => {
    await tauri.deleteObservation('obs1');
    expect(mockInvoke).toHaveBeenCalledWith('delete_observation_cmd', { observationId: 'obs1' });
  });
});

describe('confirmObservation', () => {
  it('passes observationId and confirmed', async () => {
    await tauri.confirmObservation('obs1', true);
    expect(mockInvoke).toHaveBeenCalledWith('confirm_observation_cmd', { observationId: 'obs1', confirmed: true });
  });
});

describe('getMemoryStats', () => {
  it('calls invoke', async () => {
    mockInvoke.mockResolvedValue({ total: 0, new_today: 0, confirmed: 0, domains: [] });
    await tauri.getMemoryStats();
    expect(mockInvoke).toHaveBeenCalledWith('get_memory_stats_cmd');
  });
});

describe('page domain compatibility', () => {
  const pageFromWenlanTypes = {
    id: 'page-1',
    title: 'Page',
    summary: null,
    content: 'Body',
    entity_id: null,
    space: 'work',
    source_memory_ids: [],
    version: 1,
    status: 'active',
    created_at: '2026-06-25T00:00:00Z',
    last_compiled: '2026-06-25T00:00:00Z',
    last_modified: '2026-06-25T00:00:00Z',
  };

  it('maps getPage space to domain', async () => {
    mockInvoke.mockResolvedValue(pageFromWenlanTypes);
    const page = await tauri.getPage('page-1');
    expect(mockInvoke).toHaveBeenCalledWith('get_page', { id: 'page-1' });
    expect(page?.domain).toBe('work');
  });

  it('maps searchPages space to domain', async () => {
    mockInvoke.mockResolvedValue([pageFromWenlanTypes]);
    const pages = await tauri.searchPages('query', 3);
    expect(mockInvoke).toHaveBeenCalledWith('search_pages', { query: 'query', limit: 3 });
    expect(pages[0].domain).toBe('work');
  });

  it('maps listPages space to domain', async () => {
    mockInvoke.mockResolvedValue([pageFromWenlanTypes]);
    const pages = await tauri.listPages('active', 'work', 10, 2);
    expect(mockInvoke).toHaveBeenCalledWith('list_pages', {
      status: 'active',
      domain: 'work',
      limit: 10,
      offset: 2,
    });
    expect(pages[0].domain).toBe('work');
  });

  it('creates an authored Page with normalized content and optional Space', async () => {
    mockInvoke.mockResolvedValue({ id: 'page-new', attached_to: null });

    const result = await tauri.createPage({
      title: '  A durable note  ',
      content: '  Source-backed body  ',
      space: '   ',
    });

    expect(mockInvoke).toHaveBeenCalledWith('create_page', {
      title: 'A durable note',
      content: 'Source-backed body',
      space: null,
    });
    expect(result).toEqual({ id: 'page-new', attached_to: null, warnings: [] });
  });

  it('deletes a Page through the exact desktop command contract', async () => {
    await tauri.deletePage('page-delete-me');

    expect(mockInvoke).toHaveBeenCalledWith('delete_page', { id: 'page-delete-me' });
  });
});

describe('Page draft lifecycle wrappers', () => {
  const draftPage: tauri.Page = {
    id: 'draft-1',
    title: '  Working title  ',
    summary: null,
    content: '  body whitespace stays  \n',
    entity_id: null,
    domain: null,
    space: null,
    source_memory_ids: [],
    version: 3,
    status: 'draft',
    created_at: '2026-07-16T00:00:00Z',
    last_compiled: '2026-07-16T00:00:00Z',
    last_modified: '2026-07-16T00:00:00Z',
  };

  it('creates a partial draft without trimming title or body and normalizes blank Space', async () => {
    mockInvoke.mockResolvedValue(draftPage);

    await expect(tauri.createPageDraft({
      clientDraftId: 'page_client-1',
      title: '  Working title  ',
      content: '  body whitespace stays  \n',
      space: '   ',
    })).resolves.toMatchObject({ id: 'draft-1', domain: null });

    expect(mockInvoke).toHaveBeenCalledWith('create_page_draft', {
      clientDraftId: 'page_client-1',
      title: '  Working title  ',
      content: '  body whitespace stays  \n',
      space: null,
    });
  });

  it('updates through Tauri camelCase args while preserving draft whitespace', async () => {
    mockInvoke.mockResolvedValue({ ...draftPage, version: 4, space: 'Wenlan' });

    await tauri.updatePageDraft({
      id: 'draft-1',
      expectedVersion: 3,
      title: '  revised  ',
      content: '\n  revised body  \n',
      space: '  Wenlan  ',
    });

    expect(mockInvoke).toHaveBeenCalledWith('update_page_draft', {
      id: 'draft-1',
      expectedVersion: 3,
      title: '  revised  ',
      content: '\n  revised body  \n',
      space: 'Wenlan',
    });
  });

  it('publishes and discards through versioned camelCase args', async () => {
    mockInvoke.mockResolvedValueOnce({ ...draftPage, version: 4, status: 'active' });
    await expect(tauri.publishPageDraft({ id: 'draft-1', expectedVersion: 3 }))
      .resolves.toMatchObject({ id: 'draft-1', status: 'active' });
    expect(mockInvoke).toHaveBeenLastCalledWith('publish_page_draft', {
      id: 'draft-1',
      expectedVersion: 3,
    });

    mockInvoke.mockResolvedValueOnce(undefined);
    await expect(tauri.discardPageDraft({ id: 'draft-1', expectedVersion: 4 }))
      .resolves.toBeUndefined();
    expect(mockInvoke).toHaveBeenLastCalledWith('discard_page_draft', {
      id: 'draft-1',
      expectedVersion: 4,
    });
  });

  it.each([
    [
      '{"code":"draft_version_conflict","error":"stale","current_version":7}',
      { code: 'draft_version_conflict', currentVersion: 7 },
    ],
    [
      new Error('HTTP POST /api/pages/drafts/draft-1/publish returned 409 Conflict: {"code":"page_title_conflict","error":"exists","existing_page_id":"page-9","existing_page_title":"Existing"}'),
      {
        code: 'page_title_conflict',
        existingPageId: 'page-9',
        existingPageTitle: 'Existing',
      },
    ],
    [
      '{"code":"page_draft_not_found","error":"gone"}',
      { code: 'page_draft_not_found' },
    ],
    [
      '{"code":"page_draft_id_conflict","error":"collision"}',
      { code: 'page_draft_id_conflict' },
    ],
  ])('preserves structured daemon conflicts from a string/Error.message boundary', async (rejection, expected) => {
    mockInvoke.mockRejectedValue(rejection);

    const promise = tauri.publishPageDraft({ id: 'draft-1', expectedVersion: 3 });

    await expect(promise).rejects.toBeInstanceOf(tauri.PageDraftApiError);
    await expect(promise).rejects.toMatchObject(expected);
  });
});

describe('page links wrappers', () => {
  it('getPageLinks invokes the daemon page link command', async () => {
    mockInvoke.mockResolvedValue({ outbound: [], inbound: [] });

    await tauri.getPageLinks('page-1');

    expect(mockInvoke).toHaveBeenCalledWith('get_page_links', { pageId: 'page-1' });
  });

  it('listOrphanLinks invokes the daemon orphan link command with minCount', async () => {
    mockInvoke.mockResolvedValue({ min_count: 2, orphan_labels: [] });

    await tauri.listOrphanLinks(2);

    expect(mockInvoke).toHaveBeenCalledWith('list_orphan_links', { minCount: 2 });
  });
});

describe('distillReview', () => {
  it('distillReview invokes the review command without target or force args', async () => {
    const payload: tauri.DistillReviewResponse = {
      pages_created: 0,
      scoped: false,
      created_ids: [],
      pending: [],
      stale_pages: [],
      stale_truncated: false,
      orphan_topics: [],
    };
    mockInvoke.mockResolvedValue(payload);

    const result = await tauri.distillReview();

    expect(result).toEqual(payload);
    expect(mockInvoke).toHaveBeenCalledWith('distill_review');
  });
});

describe('redistillPage', () => {
  it('invokes the page-specific re-distill command with only pageId', async () => {
    const payload: tauri.PageRedistillResponse = {
      status: 'skipped',
      updated: false,
      hint: 'page re-distill needs an LLM in the daemon',
    };
    mockInvoke.mockResolvedValue(payload);

    const result = await tauri.redistillPage('page-refresh');

    expect(result).toEqual(payload);
    expect(mockInvoke).toHaveBeenCalledWith('redistill_page', { pageId: 'page-refresh' });
  });
});

describe("daemonMeetsFloor", () => {
  it("rejects the live 0.9.5 daemon (below floor)", () => {
    expect(daemonMeetsFloor("0.9.5")).toBe(false);
  });
  it("accepts exactly the floor and above", () => {
    expect(daemonMeetsFloor("0.10.0")).toBe(true);
    expect(daemonMeetsFloor("0.10.1")).toBe(true);
    expect(daemonMeetsFloor("0.11.0")).toBe(true);
    expect(daemonMeetsFloor("1.0.0")).toBe(true);
  });
  it("tolerates a pre-release suffix", () => {
    expect(daemonMeetsFloor("0.11.0-rc.1")).toBe(true);
  });
});
