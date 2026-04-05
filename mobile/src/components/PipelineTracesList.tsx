import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  FlatList,
  View,
  Text,
  Pressable,
  ActivityIndicator,
  StyleSheet,
} from "react-native";
import { createApiClient, getDefaultApiBaseUrl } from "../../shared/core/api";
import { PipelineTrace, TraceAnalytics } from "../../shared/core/types";
import TraceListItem from "./TraceListItem";
import TraceDetailModal from "./TraceDetailModal";
import DateRangeFilter from "./DateRangeFilter";
import { NEURAL, SEMANTIC_COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from "../theme/colors";

interface PipelineTracesListProps {
  baseUrl?: string;
  refreshInterval?: number; // ms, 0 = disabled
  maxTraces?: number; // 10, 20, 50, 100
}

export default function PipelineTracesList({
  baseUrl,
  refreshInterval = 10000,
  maxTraces = 20,
}: PipelineTracesListProps) {
  const api = useMemo(
    () => createApiClient({ baseUrl: baseUrl || getDefaultApiBaseUrl() }),
    [baseUrl]
  );
  const [traces, setTraces] = useState<PipelineTrace[]>([]);
  const [analytics, setAnalytics] = useState<TraceAnalytics | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedTrace, setSelectedTrace] = useState<PipelineTrace | null>(null);
  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [filteredTraces, setFilteredTraces] = useState<PipelineTrace[]>([]);

  const [startTime, setStartTime] = useState<number | null>(null);
  const [endTime, setEndTime] = useState<number | null>(null);
  const [showFilter, setShowFilter] = useState(false);

  const applyFilters = useCallback(
    (items: PipelineTrace[], start: number | null, end: number | null) => {
      let filtered = [...items];

      if (start !== null || end !== null) {
        filtered = filtered.filter((trace) => {
          const timestamp = new Date(trace.timestamp).getTime();
          if (start !== null && timestamp < start) {
            return false;
          }
          if (end !== null && timestamp > end) {
            return false;
          }
          return true;
        });
      }

      setFilteredTraces(filtered);
    },
    []
  );

  const loadTraces = useCallback(async () => {
    try {
      setLoading(true);
      const res = await api.getPipelineTraces(maxTraces);
      setTraces(res.traces || []);
      setAnalytics(res.analytics || null);
      applyFilters(res.traces || [], startTime, endTime);
    } catch (err) {
      console.error("Failed to load pipeline traces:", err);
      setTraces([]);
    } finally {
      setLoading(false);
    }
  }, [api, applyFilters, endTime, maxTraces, startTime]);

  useEffect(() => {
    loadTraces();

    if (refreshInterval > 0) {
      const interval = setInterval(loadTraces, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [loadTraces, refreshInterval]);

  const handleFilterApply = (start: number | null, end: number | null) => {
    setStartTime(start);
    setEndTime(end);
    applyFilters(traces, start, end);
    setShowFilter(false);
  };

  const handleTracePress = (trace: PipelineTrace) => {
    setSelectedTrace(trace);
    setDetailModalVisible(true);
  };

  const handleCloseDetail = () => {
    setDetailModalVisible(false);
    setSelectedTrace(null);
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Pipeline Traces</Text>
        <View style={styles.headerControls}>
          <Pressable
            style={[
              styles.filterButton,
              showFilter && styles.filterButtonActive,
            ]}
            onPress={() => setShowFilter(!showFilter)}
          >
            <Text style={styles.filterButtonText}>
              {showFilter ? "Hide Filter" : "Filter"}
            </Text>
          </Pressable>
          {filteredTraces.length !== traces.length ? (
            <Text style={styles.filterIndicator}>
              {filteredTraces.length}/{traces.length}
            </Text>
          ) : null}
        </View>
      </View>

      {showFilter ? (
        <View style={styles.filterContainer}>
          <DateRangeFilter
            onApply={handleFilterApply}
            onCancel={() => setShowFilter(false)}
          />
        </View>
      ) : null}

      {analytics ? (
        <View style={styles.analyticsContainer}>
          <View style={styles.analyticsRow}>
            <View style={styles.analyticItem}>
              <Text style={styles.analyticLabel}>Avg Latency</Text>
              <Text style={styles.analyticValue}>{Math.round(analytics.avg_duration_ms || 0)}ms</Text>
            </View>
            <View style={styles.analyticItem}>
              <Text style={styles.analyticLabel}>Cache Hit</Text>
              <Text style={styles.analyticValue}>{Math.round((analytics.cache_hit_rate || 0) * 100)}%</Text>
            </View>
            <View style={styles.analyticItem}>
              <Text style={styles.analyticLabel}>Avg Confidence</Text>
              <Text style={styles.analyticValue}>{Math.round((analytics.avg_confidence || 0) * 100)}%</Text>
            </View>
          </View>
        </View>
      ) : null}

      <FlatList
        style={styles.listContainer}
        data={filteredTraces}
        keyExtractor={(item, index) => item.trace_id || `${index}`}
        onRefresh={loadTraces}
        refreshing={loading}
        contentContainerStyle={styles.tracesList}
        renderItem={({ item }) => (
          <Pressable onPress={() => handleTracePress(item)}>
            <TraceListItem trace={item} />
          </Pressable>
        )}
        ListEmptyComponent={
          loading ? (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color={NEURAL.primary} />
              <Text style={styles.loadingText}>Loading traces...</Text>
            </View>
          ) : (
            <View style={styles.emptyContainer}>
              <Text style={styles.emptyText}>No traces found</Text>
              <Text style={styles.emptySubtext}>
                {startTime || endTime
                  ? "Try adjusting your filters"
                  : "Run a chat query to see pipeline traces"}
              </Text>
            </View>
          )
        }
      />

      {selectedTrace ? (
        <TraceDetailModal
          trace={selectedTrace}
          visible={detailModalVisible}
          onClose={handleCloseDetail}
        />
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: SEMANTIC_COLORS.bgCanvas,
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.md,
    borderBottomWidth: 1,
    borderBottomColor: SEMANTIC_COLORS.borderPrimary,
    backgroundColor: SEMANTIC_COLORS.bgPrimary,
  },
  headerTitle: {
    fontSize: TYPOGRAPHY.fontSize.md,
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
    color: SEMANTIC_COLORS.textPrimary,
  },
  headerControls: {
    flexDirection: "row",
    alignItems: "center",
    gap: SPACING.md,
  },
  filterButton: {
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    borderRadius: BORDER_RADIUS.md,
    backgroundColor: SEMANTIC_COLORS.bgSecondary,
    borderWidth: 1,
    borderColor: SEMANTIC_COLORS.borderPrimary,
  },
  filterButtonActive: {
    backgroundColor: `${NEURAL.primary}18`,
    borderColor: `${NEURAL.primary}60`,
  },
  filterButtonText: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    fontWeight: TYPOGRAPHY.fontWeight.medium,
    color: SEMANTIC_COLORS.textSecondary,
  },
  filterIndicator: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: SEMANTIC_COLORS.textTertiary,
    fontWeight: TYPOGRAPHY.fontWeight.medium,
  },
  filterContainer: {
    backgroundColor: SEMANTIC_COLORS.bgPrimary,
    borderBottomWidth: 1,
    borderBottomColor: SEMANTIC_COLORS.borderPrimary,
    padding: SPACING.md,
  },
  analyticsContainer: {
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.md,
    backgroundColor: SEMANTIC_COLORS.bgPrimary,
    marginBottom: SPACING.sm,
  },
  analyticsRow: {
    flexDirection: "row",
    justifyContent: "space-around",
  },
  analyticItem: {
    alignItems: "center",
    paddingVertical: SPACING.sm,
  },
  analyticLabel: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: SEMANTIC_COLORS.textSecondary,
    marginBottom: SPACING.xs,
  },
  analyticValue: {
    fontSize: TYPOGRAPHY.fontSize.sm,
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
    color: SEMANTIC_COLORS.textPrimary,
  },
  listContainer: {
    flex: 1,
  },
  tracesList: {
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    paddingBottom: SPACING["3xl"],
  },
  loadingContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    paddingVertical: SPACING["4xl"],
  },
  loadingText: {
    fontSize: TYPOGRAPHY.fontSize.sm,
    color: SEMANTIC_COLORS.textSecondary,
    marginTop: SPACING.sm,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    paddingVertical: SPACING["5xl"],
  },
  emptyText: {
    fontSize: TYPOGRAPHY.fontSize.md,
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
    color: SEMANTIC_COLORS.textPrimary,
    marginBottom: SPACING.xs,
  },
  emptySubtext: {
    fontSize: TYPOGRAPHY.fontSize.sm,
    color: SEMANTIC_COLORS.textSecondary,
  },
});
