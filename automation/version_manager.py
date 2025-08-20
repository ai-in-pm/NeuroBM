#!/usr/bin/env python3
"""
Version Management System for NeuroBM.

This system handles semantic versioning, changelog generation, release
preparation, and deployment coordination for the automated update system.

Features:
- Semantic versioning (MAJOR.MINOR.PATCH)
- Automated changelog generation
- Release branch management
- Migration guide creation
- Rollback capabilities
- Release validation and testing

Usage:
    python automation/version_manager.py --prepare-release --version=1.2.0
    python automation/version_manager.py --generate-changelog
    python automation/version_manager.py --create-migration-guide
    python automation/version_manager.py --rollback --to-version=1.1.0
"""

import logging
import json
import yaml
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import subprocess
import git
from dataclasses import dataclass, asdict
try:
    from semantic_version import Version
except ImportError:
    # Fallback implementation if semantic_version not available
    class Version:
        def __init__(self, version_str):
            parts = version_str.split('.')
            self.major = int(parts[0]) if len(parts) > 0 else 0
            self.minor = int(parts[1]) if len(parts) > 1 else 0
            self.patch = int(parts[2]) if len(parts) > 2 else 0

        def __str__(self):
            return f"{self.major}.{self.minor}.{self.patch}"

        def next_major(self):
            return Version(f"{self.major + 1}.0.0")

        def next_minor(self):
            return Version(f"{self.major}.{self.minor + 1}.0")

        def next_patch(self):
            return Version(f"{self.major}.{self.minor}.{self.patch + 1}")
import jinja2

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReleaseInfo:
    """Data class for release information."""
    version: str
    release_date: datetime
    release_type: str  # major, minor, patch
    changes: List[Dict[str, Any]]
    breaking_changes: List[str]
    migration_required: bool
    rollback_supported: bool
    test_results: Dict[str, bool]
    
    def __post_init__(self):
        if not self.changes:
            self.changes = []
        if not self.breaking_changes:
            self.breaking_changes = []
        if not self.test_results:
            self.test_results = {}


class VersionManager:
    """Manages versioning and releases for NeuroBM."""
    
    def __init__(self, config_path: str = "automation/config/version_config.yaml"):
        """Initialize the version manager."""
        self.config = self._load_config(config_path)
        self.repo = git.Repo('.')
        self.current_version = self._get_current_version()
        self.release_history = self._load_release_history()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load version management configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {
                'versioning': {
                    'scheme': 'semantic',
                    'auto_increment': True,
                    'release_schedule': 'weekly',
                    'release_day': 'tuesday'
                },
                'changelog': {
                    'auto_generate': True,
                    'include_commits': True,
                    'include_issues': True,
                    'group_by_type': True
                },
                'release': {
                    'require_tests': True,
                    'require_documentation': True,
                    'require_migration_guide': True,
                    'auto_deploy': False
                },
                'rollback': {
                    'max_rollback_versions': 5,
                    'require_approval': True,
                    'backup_before_rollback': True
                }
            }
    
    def _get_current_version(self) -> Version:
        """Get current version from version file."""
        version_file = Path('VERSION')
        if version_file.exists():
            version_str = version_file.read_text().strip()
            return Version(version_str)
        return Version('1.0.0')
    
    def _load_release_history(self) -> List[ReleaseInfo]:
        """Load release history."""
        history_file = Path('automation/data/release_history.json')
        if history_file.exists():
            with open(history_file, 'r') as f:
                history_data = json.load(f)
                return [ReleaseInfo(**release) for release in history_data]
        return []
    
    def _save_release_history(self):
        """Save release history."""
        history_file = Path('automation/data/release_history.json')
        history_file.parent.mkdir(parents=True, exist_ok=True)
        
        history_data = []
        for release in self.release_history:
            release_dict = asdict(release)
            release_dict['release_date'] = release.release_date.isoformat()
            history_data.append(release_dict)
        
        with open(history_file, 'w') as f:
            json.dump(history_data, f, indent=2)
    
    def determine_next_version(self, changes: List[Dict[str, Any]]) -> Tuple[Version, str]:
        """Determine next version based on changes."""
        current = self.current_version
        
        # Analyze changes to determine version bump type
        has_breaking = any(change.get('breaking', False) for change in changes)
        has_features = any(change.get('type') in ['feat', 'feature'] for change in changes)
        has_fixes = any(change.get('type') in ['fix', 'bugfix'] for change in changes)
        
        if has_breaking:
            next_version = current.next_major()
            release_type = 'major'
        elif has_features:
            next_version = current.next_minor()
            release_type = 'minor'
        elif has_fixes:
            next_version = current.next_patch()
            release_type = 'patch'
        else:
            # Default to patch for any changes
            next_version = current.next_patch()
            release_type = 'patch'
        
        return next_version, release_type
    
    def prepare_release(self, target_version: Optional[str] = None) -> ReleaseInfo:
        """Prepare a new release."""
        logger.info("Preparing new release...")
        
        # Get changes since last release
        changes = self._get_changes_since_last_release()
        
        # Determine version
        if target_version:
            next_version = Version(target_version)
            release_type = self._determine_release_type(self.current_version, next_version)
        else:
            next_version, release_type = self.determine_next_version(changes)
        
        # Check for breaking changes
        breaking_changes = [
            change['description'] for change in changes 
            if change.get('breaking', False)
        ]
        
        # Create release info
        release_info = ReleaseInfo(
            version=str(next_version),
            release_date=datetime.now(),
            release_type=release_type,
            changes=changes,
            breaking_changes=breaking_changes,
            migration_required=len(breaking_changes) > 0,
            rollback_supported=True,
            test_results={}
        )
        
        # Create release branch
        self._create_release_branch(release_info)
        
        # Update version files
        self._update_version_files(next_version)
        
        # Generate changelog
        self._generate_changelog(release_info)
        
        # Generate migration guide if needed
        if release_info.migration_required:
            self._generate_migration_guide(release_info)
        
        # Run release tests
        release_info.test_results = self._run_release_tests()
        
        # Store release info
        self.release_history.append(release_info)
        self._save_release_history()
        
        logger.info(f"Release {next_version} prepared successfully")
        return release_info
    
    def _get_changes_since_last_release(self) -> List[Dict[str, Any]]:
        """Get changes since the last release."""
        changes = []
        
        # Get commits since last release
        try:
            last_tag = self.repo.git.describe('--tags', '--abbrev=0')
            commits = list(self.repo.iter_commits(f'{last_tag}..HEAD'))
        except:
            # No previous tags, get all commits
            commits = list(self.repo.iter_commits('HEAD'))
        
        for commit in commits:
            change = self._parse_commit_message(commit.message)
            if change:
                change['commit_hash'] = commit.hexsha[:8]
                change['author'] = commit.author.name
                change['date'] = commit.committed_datetime.isoformat()
                changes.append(change)
        
        # Get integration candidates that were merged
        integration_changes = self._get_integration_changes()
        changes.extend(integration_changes)
        
        return changes
    
    def _parse_commit_message(self, message: str) -> Optional[Dict[str, Any]]:
        """Parse commit message for change information."""
        # Parse conventional commit format
        pattern = r'^(feat|fix|docs|style|refactor|test|chore|integration)(\(.+\))?: (.+)'
        match = re.match(pattern, message.strip())
        
        if match:
            change_type, scope, description = match.groups()
            
            return {
                'type': change_type,
                'scope': scope.strip('()') if scope else None,
                'description': description,
                'breaking': 'BREAKING CHANGE' in message or '!' in (scope or ''),
                'raw_message': message
            }
        
        # Fallback for non-conventional commits
        return {
            'type': 'other',
            'scope': None,
            'description': message.split('\n')[0][:100],
            'breaking': False,
            'raw_message': message
        }
    
    def _get_integration_changes(self) -> List[Dict[str, Any]]:
        """Get changes from research integrations."""
        changes = []
        
        # Load recent integration candidates
        candidates_dir = Path('automation/data/integration_candidates')
        if candidates_dir.exists():
            for candidates_file in candidates_dir.glob('candidates_*.json'):
                try:
                    with open(candidates_file, 'r') as f:
                        candidates = json.load(f)
                    
                    for candidate in candidates:
                        # Check if this candidate was integrated
                        if self._was_candidate_integrated(candidate):
                            changes.append({
                                'type': 'integration',
                                'scope': candidate['integration_type'],
                                'description': f"Integrated research: {candidate['title'][:50]}...",
                                'breaking': candidate.get('breaking_changes', False),
                                'paper_id': candidate['paper_id']
                            })
                except Exception as e:
                    logger.warning(f"Could not process candidates file {candidates_file}: {e}")
        
        return changes
    
    def _was_candidate_integrated(self, candidate: Dict[str, Any]) -> bool:
        """Check if an integration candidate was actually integrated."""
        # Look for integration commits or files
        paper_id = candidate['paper_id']
        
        # Check for commits mentioning the paper ID
        try:
            commits = list(self.repo.iter_commits('--grep', paper_id, 'HEAD'))
            return len(commits) > 0
        except:
            return False
    
    def _determine_release_type(self, current: Version, next_version: Version) -> str:
        """Determine release type from version comparison."""
        if next_version.major > current.major:
            return 'major'
        elif next_version.minor > current.minor:
            return 'minor'
        else:
            return 'patch'
    
    def _create_release_branch(self, release_info: ReleaseInfo):
        """Create release branch."""
        branch_name = f"release/{release_info.version}"
        
        try:
            # Ensure we're on development branch
            dev_branch = 'dev'  # From config
            self.repo.git.checkout(dev_branch)
            
            # Create release branch
            self.repo.git.checkout('-b', branch_name)
            logger.info(f"Created release branch: {branch_name}")
            
        except Exception as e:
            logger.error(f"Failed to create release branch: {e}")
            raise
    
    def _update_version_files(self, version: Version):
        """Update version in all relevant files."""
        # Update VERSION file
        version_file = Path('VERSION')
        with open(version_file, 'w') as f:
            f.write(str(version))
        
        # Update pyproject.toml
        pyproject_file = Path('pyproject.toml')
        if pyproject_file.exists():
            content = pyproject_file.read_text()
            content = re.sub(
                r'version = "[^"]*"',
                f'version = "{version}"',
                content
            )
            pyproject_file.write_text(content)
        
        # Update __init__.py
        init_file = Path('neurobm/__init__.py')
        if init_file.exists():
            content = init_file.read_text()
            content = re.sub(
                r'__version__ = "[^"]*"',
                f'__version__ = "{version}"',
                content
            )
            init_file.write_text(content)
        
        logger.info(f"Updated version files to {version}")
    
    def _generate_changelog(self, release_info: ReleaseInfo):
        """Generate changelog for the release."""
        changelog_file = Path('CHANGELOG.md')
        
        # Read existing changelog
        if changelog_file.exists():
            existing_content = changelog_file.read_text()
        else:
            existing_content = "# Changelog\n\nAll notable changes to this project will be documented in this file.\n\n"
        
        # Generate new entry
        new_entry = self._format_changelog_entry(release_info)
        
        # Insert new entry
        lines = existing_content.split('\n')
        
        # Find insertion point (after header)
        insert_index = 3  # After title and description
        for i, line in enumerate(lines[3:], 3):
            if line.startswith('## '):
                insert_index = i
                break
        
        # Insert new entry
        new_lines = lines[:insert_index] + new_entry.split('\n') + [''] + lines[insert_index:]
        
        # Write updated changelog
        with open(changelog_file, 'w') as f:
            f.write('\n'.join(new_lines))
        
        logger.info("Generated changelog entry")
    
    def _format_changelog_entry(self, release_info: ReleaseInfo) -> str:
        """Format changelog entry for release."""
        entry = f"## [{release_info.version}] - {release_info.release_date.strftime('%Y-%m-%d')}\n"
        
        # Group changes by type
        change_groups = {}
        for change in release_info.changes:
            change_type = change['type']
            if change_type not in change_groups:
                change_groups[change_type] = []
            change_groups[change_type].append(change)
        
        # Format each group
        type_headers = {
            'feat': 'Added',
            'fix': 'Fixed',
            'docs': 'Documentation',
            'integration': 'Research Integrations',
            'other': 'Other'
        }
        
        for change_type, changes in change_groups.items():
            if not changes:
                continue
                
            header = type_headers.get(change_type, change_type.title())
            entry += f"\n### {header}\n"
            
            for change in changes:
                description = change['description']
                if change.get('scope'):
                    description = f"**{change['scope']}**: {description}"
                
                entry += f"- {description}\n"
                
                if change.get('paper_id'):
                    entry += f"  - Research integration: {change['paper_id']}\n"
        
        # Add breaking changes section
        if release_info.breaking_changes:
            entry += "\n### Breaking Changes\n"
            for breaking_change in release_info.breaking_changes:
                entry += f"- {breaking_change}\n"
        
        # Add migration note
        if release_info.migration_required:
            entry += f"\n**Migration Required**: See migration guide for v{release_info.version}\n"
        
        return entry
    
    def _generate_migration_guide(self, release_info: ReleaseInfo):
        """Generate migration guide for breaking changes."""
        migration_dir = Path('docs/migrations')
        migration_dir.mkdir(parents=True, exist_ok=True)
        
        migration_file = migration_dir / f"v{release_info.version}.md"
        
        template = f"""# Migration Guide: v{release_info.version}

This guide helps you migrate from previous versions to v{release_info.version}.

## Overview

This release includes breaking changes that require migration steps.

**Release Type**: {release_info.release_type}
**Migration Required**: Yes
**Estimated Migration Time**: 15-30 minutes

## Breaking Changes

"""
        
        for breaking_change in release_info.breaking_changes:
            template += f"- {breaking_change}\n"
        
        template += """

## Migration Steps

### 1. Backup Your Current Setup

Before starting the migration, create a backup of your current NeuroBM installation:

```bash
# Create backup
cp -r /path/to/neurobm /path/to/neurobm_backup

# Or use git to create a backup branch
git checkout -b backup_pre_v{version}
git push origin backup_pre_v{version}
```

### 2. Update Dependencies

Update your Python dependencies:

```bash
pip install --upgrade neurobm
```

### 3. Configuration Changes

[Specific configuration changes will be listed here based on the actual breaking changes]

### 4. Code Changes

[Specific code changes will be listed here based on the actual breaking changes]

### 5. Test Your Setup

After migration, run the validation tests:

```bash
python tools/validate_project.py
python test_neurobm_comprehensive.py
```

## Rollback Instructions

If you encounter issues, you can rollback to the previous version:

```bash
python automation/version_manager.py --rollback --to-version={previous_version}
```

## Support

If you encounter issues during migration:

1. Check the troubleshooting section below
2. Review the changelog for additional details
3. Create an issue on GitHub with migration details

## Troubleshooting

### Common Issues

[Common migration issues and solutions will be added here]

---

**Note**: This migration guide was automatically generated. Please review all changes carefully and test thoroughly in a development environment before applying to production.
""".format(version=release_info.version, previous_version=str(self.current_version))
        
        with open(migration_file, 'w') as f:
            f.write(template)
        
        logger.info(f"Generated migration guide: {migration_file}")
    
    def _run_release_tests(self) -> Dict[str, bool]:
        """Run comprehensive tests for the release."""
        test_results = {}
        
        # Run unit tests
        try:
            result = subprocess.run(['python', '-m', 'pytest', 'tests/'], 
                                  capture_output=True, text=True, timeout=600)
            test_results['unit_tests'] = result.returncode == 0
        except Exception as e:
            logger.error(f"Unit tests failed: {e}")
            test_results['unit_tests'] = False
        
        # Run integration tests
        try:
            result = subprocess.run(['python', 'test_neurobm_comprehensive.py'], 
                                  capture_output=True, text=True, timeout=300)
            test_results['integration_tests'] = result.returncode == 0
        except Exception as e:
            logger.error(f"Integration tests failed: {e}")
            test_results['integration_tests'] = False
        
        # Run project validation
        try:
            result = subprocess.run(['python', 'tools/validate_project.py'], 
                                  capture_output=True, text=True, timeout=120)
            test_results['project_validation'] = result.returncode == 0
        except Exception as e:
            logger.error(f"Project validation failed: {e}")
            test_results['project_validation'] = False
        
        # Check documentation
        test_results['documentation_complete'] = self._check_documentation_completeness()
        
        return test_results
    
    def _check_documentation_completeness(self) -> bool:
        """Check if documentation is complete."""
        required_docs = [
            'README.md',
            'CHANGELOG.md',
            'docs/ethics_guidelines.md',
            'docs/responsible_ai_framework.md'
        ]
        
        for doc in required_docs:
            if not Path(doc).exists():
                logger.warning(f"Missing documentation: {doc}")
                return False
        
        return True
    
    def finalize_release(self, release_info: ReleaseInfo) -> bool:
        """Finalize and tag the release."""
        try:
            # Commit all release changes
            self.repo.git.add('.')
            commit_message = f"chore: prepare release v{release_info.version}"
            self.repo.git.commit('-m', commit_message)
            
            # Create and push tag
            tag_name = f"v{release_info.version}"
            self.repo.create_tag(tag_name, message=f"Release {release_info.version}")
            
            # Merge to staging branch
            staging_branch = 'staging'
            self.repo.git.checkout(staging_branch)
            self.repo.git.merge(f"release/{release_info.version}")
            
            logger.info(f"Release v{release_info.version} finalized and tagged")
            return True
            
        except Exception as e:
            logger.error(f"Failed to finalize release: {e}")
            return False
    
    def rollback_to_version(self, target_version: str) -> bool:
        """Rollback to a specific version."""
        logger.info(f"Rolling back to version {target_version}")
        
        try:
            # Validate target version exists
            tag_name = f"v{target_version}"
            if tag_name not in [tag.name for tag in self.repo.tags]:
                logger.error(f"Version {target_version} not found in tags")
                return False
            
            # Create backup before rollback
            backup_branch = f"backup_before_rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.repo.git.checkout('-b', backup_branch)
            self.repo.git.checkout('main')
            
            # Reset to target version
            self.repo.git.reset('--hard', tag_name)
            
            # Update version files
            self._update_version_files(Version(target_version))
            
            # Commit rollback
            self.repo.git.add('.')
            self.repo.git.commit('-m', f"rollback: revert to version {target_version}")
            
            logger.info(f"Successfully rolled back to version {target_version}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False


def main():
    """Main function for version management."""
    import argparse
    
    parser = argparse.ArgumentParser(description='NeuroBM Version Manager')
    parser.add_argument('--prepare-release', action='store_true', help='Prepare new release')
    parser.add_argument('--version', type=str, help='Target version for release')
    parser.add_argument('--finalize-release', action='store_true', help='Finalize prepared release')
    parser.add_argument('--generate-changelog', action='store_true', help='Generate changelog')
    parser.add_argument('--rollback', action='store_true', help='Rollback to version')
    parser.add_argument('--to-version', type=str, help='Target version for rollback')
    
    args = parser.parse_args()
    
    manager = VersionManager()
    
    if args.prepare_release:
        release_info = manager.prepare_release(args.version)
        print(f"Prepared release v{release_info.version}")
        print(f"Release type: {release_info.release_type}")
        print(f"Changes: {len(release_info.changes)}")
        print(f"Breaking changes: {len(release_info.breaking_changes)}")
        
        if not all(release_info.test_results.values()):
            print("⚠️ Some tests failed. Review before finalizing.")
        else:
            print("✅ All tests passed. Ready to finalize.")
    
    elif args.finalize_release:
        # Load latest release info and finalize
        if manager.release_history:
            latest_release = manager.release_history[-1]
            success = manager.finalize_release(latest_release)
            if success:
                print(f"✅ Release v{latest_release.version} finalized")
            else:
                print("❌ Release finalization failed")
        else:
            print("No release to finalize")
    
    elif args.rollback:
        if not args.to_version:
            print("Please specify --to-version for rollback")
            return
        
        success = manager.rollback_to_version(args.to_version)
        if success:
            print(f"✅ Rolled back to version {args.to_version}")
        else:
            print("❌ Rollback failed")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
